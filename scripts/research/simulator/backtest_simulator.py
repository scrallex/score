from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from stm_backtest.backtester import BacktestResult, EchoBacktester, StrategyConfig
from stm_backtest.signals import STMManifoldBuilder


@dataclass(frozen=True)
class GuardThresholds:
    """Minimum structural requirements for admitting a signal into the backtest."""

    min_repetitions: int = 3
    max_hazard: float = 0.5
    min_coherence: float = 0.5
    min_stability: float = 0.5
    max_entropy: float = 3.0


@dataclass
class InstrumentPayload:
    """Container for all inputs required to simulate a single instrument."""

    instrument: str
    candles: pd.DataFrame
    guards: GuardThresholds = field(default_factory=GuardThresholds)
    gates: Optional[pd.DataFrame] = None
    strategy_overrides: Optional[Dict[str, object]] = None


class BacktestSimulator:
    """High level façade that mirrors the live Gate → Sizer → Planner pipeline."""

    def __init__(
        self,
        *,
        base_strategy: StrategyConfig,
        builder_kwargs: Optional[Dict[str, object]] = None,
        min_trades: int = 10,
    ) -> None:
        self._base_config = base_strategy.as_dict()
        self._builder_kwargs = builder_kwargs or {}
        self._builder: Optional[STMManifoldBuilder] = None
        self._min_trades = max(1, int(min_trades))
        self._strategy_fields = set(StrategyConfig.__dataclass_fields__.keys())

    # ------------------------------------------------------------------
    def simulate_instrument(self, payload: InstrumentPayload) -> Dict[str, object]:
        source = "gates"
        signals = None
        if payload.gates is not None and not payload.gates.empty:
            signals = payload.gates.copy()
        else:
            source = "derived"
            signals = self._derive_price_signals(payload.candles, payload.instrument)

        if signals.empty:
            return self._empty_result(payload.instrument, source, 0.0)

        filtered = self._apply_guards(signals, payload.guards)
        coverage = float(len(filtered) / len(signals)) if len(signals) else 0.0
        if filtered.empty:
            return self._empty_result(payload.instrument, source, coverage)

        strategy = self._make_strategy(payload.strategy_overrides)
        runner = EchoBacktester(
            candles=payload.candles,
            signals=filtered,
            instrument=payload.instrument,
            config=strategy,
        )
        result = runner.run()
        metrics = result.summary()
        metrics = {key: self._to_python(value) for key, value in metrics.items()}
        metrics["profit_factor"] = self._clamped_profit_factor(result)
        metrics["trade_count"] = int(metrics.get("trade_count", 0))
        metrics["qualified"] = bool(metrics["trade_count"] >= self._min_trades)

        trades = self._serialize_frame(result.trade_frame())
        equity = self._serialize_frame(result.equity_curve())

        return {
            "instrument": payload.instrument,
            "metrics": metrics,
            "trades": trades,
            "equity_curve": equity,
            "gate_coverage": float(coverage),
            "source": source,
        }

    # ------------------------------------------------------------------
    def _make_strategy(self, overrides: Optional[Dict[str, object]]) -> StrategyConfig:
        params: Dict[str, object] = dict(self._base_config)
        if overrides:
            for key, value in overrides.items():
                if key not in self._strategy_fields:
                    raise ValueError(f"Unknown strategy override '{key}'")
                params[key] = value
        return StrategyConfig(**params)

    def _derive_price_signals(self, candles: pd.DataFrame, instrument: str) -> pd.DataFrame:
        builder = self._ensure_builder()
        return builder.build(candles, instrument=instrument)

    def _ensure_builder(self) -> STMManifoldBuilder:
        if self._builder is None:
            self._builder = STMManifoldBuilder(**self._builder_kwargs)
        return self._builder

    @staticmethod
    def _required_columns() -> set[str]:
        return {
            "coherence",
            "stability",
            "entropy",
            "rupture",
            "lambda_hazard",
            "repetition_count",
            "price",
            "timestamp",
            "timestamp_ms",
        }

    def _apply_guards(self, signals: pd.DataFrame, guards: GuardThresholds) -> pd.DataFrame:
        missing = self._required_columns() - set(signals.columns)
        if missing:
            missing_cols = ", ".join(sorted(missing))
            raise ValueError(f"Signals missing required columns: {missing_cols}")
        mask = (
            (signals["repetition_count"].astype(float) >= guards.min_repetitions)
            & (signals["lambda_hazard"].astype(float) <= guards.max_hazard)
            & (signals["coherence"].astype(float) >= guards.min_coherence)
            & (signals["stability"].astype(float) >= guards.min_stability)
            & (signals["entropy"].astype(float) <= guards.max_entropy)
        )
        filtered = signals.loc[mask].copy()
        filtered.reset_index(drop=True, inplace=True)
        return filtered

    def _clamped_profit_factor(self, result: BacktestResult) -> float:
        if not result.trades:
            return 0.0
        returns = np.array([trade.pnl_pct for trade in result.trades], dtype=float)
        if returns.size == 0:
            return 0.0
        wins = returns[returns > 0].sum()
        losses = -returns[returns < 0].sum()
        if losses <= 1e-6:
            losses = 1e-6
        profit_factor = wins / losses if losses > 0 else 0.0
        return float(profit_factor)

    @staticmethod
    def _serialize_frame(df: pd.DataFrame) -> List[Dict[str, object]]:
        if df.empty:
            return []
        records: List[Dict[str, object]] = []
        for raw in df.to_dict(orient="records"):
            record: Dict[str, object] = {}
            for key, value in raw.items():
                record[key] = BacktestSimulator._normalise_scalar(value)
            records.append(record)
        return records

    @staticmethod
    def _normalise_scalar(value: object) -> object:
        if value is None:
            return None
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        if isinstance(value, (np.floating,)):
            val = float(value)
            return None if math.isnan(val) or math.isinf(val) else val
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, pd.Timestamp):
            if pd.isna(value):
                return None
            ts = value
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            return ts.isoformat().replace("+00:00", "Z")
        if isinstance(value, pd.Timedelta):
            return value.total_seconds()
        return value

    @staticmethod
    def _to_python(value: object) -> object:
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return float(value)
        return value

    def _empty_result(self, instrument: str, source: str, coverage: float) -> Dict[str, object]:
        metrics = {
            "trade_count": 0,
            "avg_return_bps": 0.0,
            "win_rate": 0.0,
            "payoff": 0.0,
            "sharpe": 0.0,
            "calmar": 0.0,
            "sortino": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "avg_time_in_trade_min": 0.0,
            "alpha_vs_baseline_bps": 0.0,
            "bootstrap_p_value": None,
            "qualified": False,
        }
        return {
            "instrument": instrument,
            "metrics": metrics,
            "trades": [],
            "equity_curve": [],
            "gate_coverage": float(coverage),
            "source": source,
        }


__all__ = ["BacktestSimulator", "GuardThresholds", "InstrumentPayload"]
