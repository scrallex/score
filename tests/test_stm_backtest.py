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
