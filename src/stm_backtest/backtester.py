"""Echo backtesting engine implementing STM trading strategies."""

from __future__ import annotations

import importlib
import itertools
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scripts.trading.guards import PathMetrics, throttle_factor
except ModuleNotFoundError:  # pragma: no cover - fallback for test environments
    repo_root = Path(__file__).resolve().parents[3]
    guards_path = repo_root / "scripts" / "trading" / "guards.py"
    if not guards_path.exists():
        raise
    spec = importlib.util.spec_from_file_location("stm_backtest._guards", guards_path)
    if spec is None or spec.loader is None:
        raise
    guards_mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = guards_mod
    spec.loader.exec_module(guards_mod)
    PathMetrics = guards_mod.PathMetrics  # type: ignore[attr-defined]
    throttle_factor = guards_mod.throttle_factor  # type: ignore[attr-defined]

@dataclass(slots=True)
class StrategyConfig:
    """Configuration for a single echo trading strategy."""

    min_repetitions: int = 3
    max_hazard: float = 0.4
    min_coherence: float = 0.5
    min_stability: float = 0.5
    max_entropy: float = 3.0
    direction: str = "momentum"  # "momentum" or "mean_reversion"
    momentum_lookback: int = 3
    exit_horizon: Optional[int] = 60  # number of signals
    atr_window: int = 14
    sl_multiplier: float = 1.0
    tp_multiplier: float = 1.5
    hazard_exit_threshold: Optional[float] = 0.6
    position_mode: str = "throttle"  # or "unit"
    use_atr_targets: bool = False
    seed: Optional[int] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "min_repetitions": self.min_repetitions,
            "max_hazard": self.max_hazard,
            "min_coherence": self.min_coherence,
            "min_stability": self.min_stability,
            "max_entropy": self.max_entropy,
            "direction": self.direction,
            "momentum_lookback": self.momentum_lookback,
            "exit_horizon": self.exit_horizon,
            "atr_window": self.atr_window,
            "sl_multiplier": self.sl_multiplier,
            "tp_multiplier": self.tp_multiplier,
            "hazard_exit_threshold": self.hazard_exit_threshold,
            "position_mode": self.position_mode,
            "use_atr_targets": self.use_atr_targets,
            "seed": self.seed,
        }


@dataclass(slots=True)
class Trade:
    instrument: str
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_price: float
    exit_price: float
    direction: int  # +1 long, -1 short
    size: float
    bars_held: int
    reason: str
    entry_signal_index: int
    exit_signal_index: int
    hazard_on_exit: float
    pnl_pct: float

    @property
    def pnl_bps(self) -> float:
        return self.pnl_pct * 10_000.0


@dataclass(slots=True)
class BacktestResult:
    trades: List[Trade]
    baseline: pd.Series
    signals: pd.DataFrame
    config: StrategyConfig

    def summary(self) -> Dict[str, float | int | None]:
        if not self.trades:
            return {
                "trade_count": 0,
                "avg_return_bps": 0.0,
                "win_rate": 0.0,
                "payoff": 0.0,
                "sharpe": 0.0,
                "calmar": 0.0,
                "max_drawdown_pct": 0.0,
                "avg_time_in_trade_min": 0.0,
                "alpha_vs_baseline_bps": 0.0,
                "bootstrap_p_value": None,
            }
        returns = np.array([t.pnl_pct for t in self.trades])
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        avg_return = returns.mean() if len(returns) else 0.0
        baseline_mean = float(self.baseline.mean()) if not self.baseline.empty else 0.0
        sharpe = self._sharpe_ratio(returns)
        equity = self._equity_curve(returns)
        max_dd = self._max_drawdown(equity)
        calmar = (avg_return * 252.0) / max_dd if max_dd > 1e-9 else 0.0
        bootstrap_p = self._bootstrap_p_value(returns)
        avg_minutes = float(np.mean([t.bars_held for t in self.trades]) * 1.0)
        payoff = (wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else 0.0
        return {
            "trade_count": len(self.trades),
            "avg_return_bps": avg_return * 10_000.0,
            "win_rate": float((returns > 0).mean()),
            "payoff": float(payoff),
            "sharpe": sharpe,
            "calmar": calmar,
            "max_drawdown_pct": max_dd * 100.0,
            "avg_time_in_trade_min": avg_minutes,
            "alpha_vs_baseline_bps": (avg_return - baseline_mean) * 10_000.0,
            "bootstrap_p_value": bootstrap_p,
        }

    @staticmethod
    def _sharpe_ratio(returns: np.ndarray) -> float:
        if len(returns) < 2:
            return 0.0
        std = returns.std(ddof=1)
        if std == 0:
            return 0.0
        return returns.mean() / std * math.sqrt(len(returns))

    @staticmethod
    def _equity_curve(returns: np.ndarray) -> np.ndarray:
        return np.cumsum(returns)

    @staticmethod
    def _max_drawdown(equity: np.ndarray) -> float:
        if equity.size == 0:
            return 0.0
        peaks = np.maximum.accumulate(equity)
        drawdown = peaks - equity
        max_dd = drawdown.max()
        if peaks.max() == 0:
            return float(max_dd)
        return float(max_dd)

    @staticmethod
    def _bootstrap_p_value(returns: np.ndarray, iterations: int = 2000, seed: Optional[int] = None) -> Optional[float]:
        if len(returns) == 0:
            return None
        rng = np.random.default_rng(seed)
        mean_ret = returns.mean()
        if len(returns) == 1:
            return 0.5 if mean_ret > 0 else 1.0
        boot = []
        for _ in range(iterations):
            sample = rng.choice(returns, size=len(returns), replace=True)
            boot.append(sample.mean())
        boot = np.array(boot)
        return float(np.mean(boot <= 0.0))


class EchoBacktester:
    """Simulate echo strategies on STM signals."""

    def __init__(
        self,
        *,
        candles: pd.DataFrame,
        signals: pd.DataFrame,
        instrument: str,
        config: StrategyConfig,
    ) -> None:
        self.candles = candles.copy()
        self.signals = signals.copy()
        self.instrument = instrument
        self.config = config
        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
        self._prepare_auxiliary_columns()

    # ------------------------------------------------------------------

    def run(self) -> BacktestResult:
        trades: List[Trade] = []
        open_trade: Optional[Trade] = None
        atr_levels: Dict[int, Tuple[float, float]] = {}
        prev_metrics: Optional[PathMetrics] = None

        for idx, row in self.signals.iterrows():
            idx = int(idx)
            eligible = self._is_eligible(row)
            metrics = PathMetrics(
                entropy=float(row["entropy"]),
                coherence=float(row["coherence"]),
                stability=float(row["stability"]),
                rupture=float(row["rupture"]),
                hazard=float(row["lambda_hazard"]),
            )

            if open_trade is not None:
                should_exit, reason = self._should_exit(open_trade, row, atr_levels)
                if should_exit:
                    atr_levels.pop(open_trade.entry_signal_index, None)
                    open_trade.exit_ts = row["timestamp"]
                    open_trade.exit_price = float(row["price"])
                    open_trade.exit_signal_index = idx
                    open_trade.hazard_on_exit = float(row["lambda_hazard"])
                    open_trade.pnl_pct = open_trade.direction * (open_trade.exit_price - open_trade.entry_price) / open_trade.entry_price * open_trade.size
                    open_trade.reason = reason
                    trades.append(open_trade)
                    open_trade = None

            if open_trade is None and eligible:
                direction = self._direction(idx)
                if direction == 0:
                    prev_metrics = metrics
                    continue
                size = self._position_size(metrics, prev_metrics)
                if size <= 0:
                    prev_metrics = metrics
                    continue
                open_trade = Trade(
                    instrument=self.instrument,
                    entry_ts=row["timestamp"],
                    exit_ts=row["timestamp"],
                    entry_price=float(row["price"]),
                    exit_price=float(row["price"]),
                    direction=direction,
                    size=size,
                    bars_held=0,
                    reason="open",
                    entry_signal_index=idx,
                    exit_signal_index=idx,
                    hazard_on_exit=float(row["lambda_hazard"]),
                    pnl_pct=0.0,
                )
                if self.config.use_atr_targets:
                    atr_levels[idx] = self._atr_targets(row, direction)

            if open_trade is not None:
                open_trade.bars_held += 1

            prev_metrics = metrics

        # Flush open trade at the end
        if open_trade is not None:
            atr_levels.pop(open_trade.entry_signal_index, None)
            open_trade.exit_ts = self.signals.iloc[-1]["timestamp"]
            open_trade.exit_price = float(self.signals.iloc[-1]["price"])
            open_trade.exit_signal_index = int(self.signals.index[-1])
            open_trade.hazard_on_exit = float(self.signals.iloc[-1]["lambda_hazard"])
            open_trade.pnl_pct = open_trade.direction * (
                open_trade.exit_price - open_trade.entry_price
            ) / open_trade.entry_price * open_trade.size
            open_trade.reason = "close_eod"
            trades.append(open_trade)

        baseline = self._baseline_returns()
        return BacktestResult(trades=trades, baseline=baseline, signals=self.signals, config=self.config)

    # ------------------------------------------------------------------
    # Helpers

    def _prepare_auxiliary_columns(self) -> None:
        # Merge ATR onto signals for optional exits
        candles = self.candles.copy()
        candles.sort_values("timestamp_ms", inplace=True)
        candles["prev_close"] = candles["close"].shift(1)
        tr = pd.concat(
            [
                (candles["high"] - candles["low"]).abs(),
                (candles["high"] - candles["prev_close"]).abs(),
                (candles["low"] - candles["prev_close"]).abs(),
            ],
            axis=1,
        ).max(axis=1)
        candles["atr"] = tr.rolling(self.config.atr_window, min_periods=1).mean()
        atr = candles.set_index("timestamp_ms")["atr"]
        self.signals.sort_values("timestamp_ms", inplace=True)
        self.signals.reset_index(drop=True, inplace=True)
        self.signals["atr"] = atr.reindex(self.signals["timestamp_ms"], method="ffill").ffill().bfill()
        self.signals["price_shift"] = self.signals["price"].shift(self.config.momentum_lookback)

    def _is_eligible(self, row: pd.Series) -> bool:
        if int(row["repetition_count"]) < self.config.min_repetitions:
            return False
        if float(row["lambda_hazard"]) > self.config.max_hazard:
            return False
        if float(row["coherence"]) < self.config.min_coherence:
            return False
        if float(row["stability"]) < self.config.min_stability:
            return False
        if float(row["entropy"]) > self.config.max_entropy:
            return False
        return True

    def _direction(self, idx: int) -> int:
        row = self.signals.iloc[idx]
        ref_price = row["price_shift"]
        if pd.isna(ref_price):
            return 0
        delta = float(row["price"]) - float(ref_price)
        if abs(delta) < 1e-9:
            return 0
        if self.config.direction == "momentum":
            return 1 if delta > 0 else -1
        if self.config.direction == "mean_reversion":
            return -1 if delta > 0 else 1
        raise ValueError(f"Unknown direction mode: {self.config.direction}")

    def _position_size(self, metrics: PathMetrics, prev: Optional[PathMetrics]) -> float:
        if self.config.position_mode == "unit":
            return 1.0
        return throttle_factor(metrics, prev)

    def _should_exit(
        self,
        trade: Trade,
        row: pd.Series,
        atr_levels: Dict[int, Tuple[float, float]],
    ) -> Tuple[bool, str]:
        horizon = self.config.exit_horizon
        if horizon is not None and trade.bars_held >= horizon:
            return True, "exit_horizon"
        hz_threshold = self.config.hazard_exit_threshold
        if hz_threshold is not None and float(row["lambda_hazard"]) > hz_threshold:
            return True, "hazard_exit"
        if self.config.use_atr_targets:
            entry_idx = trade.entry_signal_index
            if entry_idx in atr_levels:
                sl, tp = atr_levels[entry_idx]
                price = float(row["price"])
                if trade.direction > 0:
                    if price <= sl:
                        return True, "stop_loss"
                    if price >= tp:
                        return True, "take_profit"
                else:
                    if price >= sl:
                        return True, "stop_loss"
                    if price <= tp:
                        return True, "take_profit"
        return False, "hold"

    def _atr_targets(self, row: pd.Series, direction: int) -> Tuple[float, float]:
        atr = float(row.get("atr", 0.0))
        if atr <= 0:
            return float(row["price"]) * 0.99, float(row["price"]) * 1.01
        entry = float(row["price"])
        sl_mult = max(1e-4, self.config.sl_multiplier)
        tp_mult = max(1e-4, self.config.tp_multiplier)
        if direction > 0:
            sl = entry - atr * sl_mult
            tp = entry + atr * tp_mult
        else:
            sl = entry + atr * sl_mult
            tp = entry - atr * tp_mult
        return sl, tp

    def _baseline_returns(self) -> pd.Series:
        if len(self.signals) < 2:
            return pd.Series(dtype=float)
        shift = self.signals["price"].shift(-1)
        delta = (shift - self.signals["price"]) / self.signals["price"]
        baseline = delta.fillna(0.0)
        directions = self.signals["price"].copy()
        directions[:] = 0
        for idx in range(len(self.signals)):
            directions.iloc[idx] = self._direction(idx)
        return baseline * directions


def parameter_grid(param_space: Dict[str, Sequence[object]]) -> Iterator[Dict[str, object]]:
    """Yield dictionaries representing the cartesian product of a parameter grid."""
    keys = list(param_space)
    values = [param_space[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def run_sweep(
    *,
    candles_by_instrument: Dict[str, pd.DataFrame],
    signals_by_instrument: Dict[str, pd.DataFrame],
    base_config: StrategyConfig,
    grid: Dict[str, Sequence[object]],
    bootstrap_iterations: int = 500,
    seed: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Run a grid sweep of strategy parameters across instruments."""
    results: List[Dict[str, object]] = []
    for params in parameter_grid(grid):
        cfg = StrategyConfig(**{**base_config.as_dict(), **params})
        all_trades: List[Trade] = []
        baseline_returns = []
        for inst, candles in candles_by_instrument.items():
            signals = signals_by_instrument[inst]
            runner = EchoBacktester(candles=candles, signals=signals, instrument=inst, config=cfg)
            res = runner.run()
            all_trades.extend(res.trades)
            baseline_returns.append(res.baseline)
        merged_signals = pd.concat(baseline_returns) if baseline_returns else pd.Series(dtype=float)
        bt = BacktestResult(trades=all_trades, baseline=merged_signals, signals=pd.DataFrame(), config=cfg)
        summary = bt.summary()
        summary.update({"instrument_count": len(candles_by_instrument), "params": params})
        summary["bootstrap_p_value"] = BacktestResult._bootstrap_p_value(
            np.array([t.pnl_pct for t in all_trades]), iterations=bootstrap_iterations, seed=seed
        )
        results.append(summary)
    return results


def dump_results(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
