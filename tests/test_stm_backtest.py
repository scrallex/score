import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stm_backtest import load_candles_csv, preprocess_candles
from stm_backtest.signals import STMManifoldBuilder
from stm_backtest.backtester import EchoBacktester, StrategyConfig


DATA_ROOT = Path(__file__).resolve().parent / "data"
SAMPLE_CSV = DATA_ROOT / "eurusd_sample.csv"


@pytest.fixture(scope="module")
def sample_candles() -> pd.DataFrame:
    raw = load_candles_csv(SAMPLE_CSV, tz="UTC")
    return preprocess_candles(raw)


def test_candle_loader_shapes(sample_candles: pd.DataFrame) -> None:
    assert not sample_candles.empty
    assert set(sample_candles.columns) >= {"timestamp", "timestamp_ms", "open", "high", "low", "close", "volume"}
    # Ensure cadence is M1
    diffs = np.unique(sample_candles["timestamp_ms"].diff().dropna())
    assert diffs.size == 1
    assert diffs[0] == pytest.approx(60_000)


def test_python_builder_matches_native(sample_candles: pd.DataFrame, tmp_path: Path) -> None:
    builder = STMManifoldBuilder(signature_precision=2, lookback_minutes=60, max_signals=None)
    python_signals = builder.build(sample_candles, instrument="EURUSD")
    assert len(python_signals) > 10

    # Prepare JSON input for native manifold generator
    json_path = tmp_path / "candles.json"
    payload = []
    for row in sample_candles.to_dict("records"):
        payload.append(
            {
                "timestamp": int(row["timestamp_ms"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
        )
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    env = os.environ.copy()
    env.update({"ECHO_SIGNATURE_PRECISION": "2", "ECHO_LOOKBACK_MINUTES": "60"})
    result = subprocess.run(
        [str(Path.cwd() / "bin" / "manifold_generator"), "--input", str(json_path)],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    native = json.loads(result.stdout)
    assert native["signals"], "native builder returned no signals"

    last_native = native["signals"][-1]
    ts_ns = int(last_native["timestamp_ns"])
    match = python_signals.loc[python_signals["timestamp_ns"] == ts_ns]
    assert not match.empty, "python builder missing matching timestamp"
    row = match.iloc[-1]
    assert row["coherence"] == pytest.approx(last_native["coherence"], rel=1e-4)
    assert row["stability"] == pytest.approx(last_native["stability"], rel=1e-4)
    assert row["entropy"] == pytest.approx(last_native["entropy"], rel=1e-4)
    assert row["lambda_hazard"] == pytest.approx(last_native["lambda_hazard"], rel=1e-4)
    assert row["signature"] == last_native["repetition"]["signature"]


def test_backtester_produces_trades(sample_candles: pd.DataFrame) -> None:
    builder = STMManifoldBuilder(signature_precision=2, lookback_minutes=60, max_signals=None)
    signals = builder.build(sample_candles, instrument="EURUSD")
    cfg = StrategyConfig(
        min_repetitions=1,
        max_hazard=0.9,
        min_coherence=0.0,
        min_stability=0.0,
        max_entropy=4.0,
        direction="momentum",
        momentum_lookback=1,
        exit_horizon=10,
        position_mode="unit",
        use_atr_targets=False,
    )
    runner = EchoBacktester(candles=sample_candles, signals=signals, instrument="EURUSD", config=cfg)
    result = runner.run()
    assert result.trades, "Expected trades to be produced"
    summary = result.summary()
    assert summary["trade_count"] == len(result.trades)
    assert isinstance(summary["avg_return_bps"], float)
    assert summary["bootstrap_p_value"] is None or 0.0 <= summary["bootstrap_p_value"] <= 1.0
    assert "profit_factor" in summary
    assert isinstance(summary["profit_factor"], float)
    assert "sortino" in summary
    assert isinstance(summary["sortino"], float)

    trades_df = result.trade_frame()
    assert not trades_df.empty
    equity_df = result.equity_curve()
    assert not equity_df.empty


def test_session_filter_blocks_entries_outside_window() -> None:
    timestamps = pd.to_datetime(
        [
            "2024-01-01T08:00:00Z",
            "2024-01-01T09:00:00Z",
            "2024-01-01T10:00:00Z",
            "2024-01-01T11:00:00Z",
        ]
    )
    timestamp_ns = timestamps.astype("int64")
    timestamp_ms = (timestamp_ns // 1_000_000).astype("int64")
    candles = pd.DataFrame(
        {
            "timestamp": timestamps,
            "timestamp_ms": timestamp_ms,
            "open": np.linspace(1.1000, 1.1030, len(timestamps)),
            "high": np.linspace(1.1005, 1.1035, len(timestamps)),
            "low": np.linspace(1.0995, 1.1025, len(timestamps)),
            "close": np.linspace(1.1000, 1.1030, len(timestamps)),
            "volume": np.linspace(1000, 1300, len(timestamps)),
        }
    )

    base_signal = {
        "instrument": "TEST",
        "coherence": 0.8,
        "stability": 0.8,
        "entropy": 0.2,
        "rupture": 0.2,
        "lambda_hazard": 0.2,
        "repetition_count": 3,
        "repetition_first_seen_ms": int(timestamp_ms[0]),
    }
    prices = [1.1000, 1.1010, 1.1020, 1.1030]
    signals = pd.DataFrame(
        {
            **{key: [value] * len(timestamps) for key, value in base_signal.items()},
            "timestamp": timestamps,
            "timestamp_ms": timestamp_ms,
            "timestamp_ns": timestamp_ns,
            "price": prices,
            "signature": [f"sig{i}" for i in range(len(timestamps))],
            "window_index": list(range(len(timestamps))),
        }
    )

    cfg = StrategyConfig(
        min_repetitions=1,
        max_hazard=0.9,
        min_coherence=0.0,
        min_stability=0.0,
        max_entropy=1.0,
        direction="momentum",
        momentum_lookback=1,
        exit_horizon=1,
        hazard_exit_threshold=None,
        position_mode="unit",
        tp_mode="OFF",
        session={"start": "09:00Z", "end": "11:00Z"},
    )

    runner = EchoBacktester(candles=candles, signals=signals, instrument="TEST", config=cfg)
    result = runner.run()

    assert result.trades, "Expected at least one trade inside the configured session"
    start_minute = 9 * 60
    end_minute = 11 * 60
    for trade in result.trades:
        minute = trade.entry_ts.hour * 60 + trade.entry_ts.minute
        assert start_minute <= minute < end_minute
        assert cfg.session.contains(int(trade.entry_ts.value // 1_000_000))
