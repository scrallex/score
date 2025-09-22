"""Echo backtesting engine implementing STM trading strategies."""

from __future__ import annotations

import importlib
import itertools
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime
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

SESSION_PRESETS: Dict[str, Tuple[str, str]] = {
    "LDN": ("07:00", "15:00"),
    "NY": ("12:00", "21:00"),
    "TKO": ("23:00", "07:00"),
    "SYD": ("21:00", "05:00"),
}


def _parse_time_string(raw: str) -> int:
    value = raw.strip().upper()
    if value.endswith("Z"):
        value = value[:-1]
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            parsed = datetime.strptime(value, fmt)
        except ValueError:
            continue
        return parsed.hour * 60 + parsed.minute
    raise ValueError(f"Invalid session time format: '{raw}'")


@dataclass(slots=True)
class SessionWindow:
    label: Optional[str]
    start_minute: int
    end_minute: int

    def contains(self, timestamp_ms: int) -> bool:
        ts = pd.to_datetime(timestamp_ms, unit="ms", utc=True)
        minute = ts.hour * 60 + ts.minute
        if self.start_minute <= self.end_minute:
            return self.start_minute <= minute < self.end_minute
        return minute >= self.start_minute or minute < self.end_minute

    def to_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "start_minute": self.start_minute,
            "end_minute": self.end_minute,
        }

    @classmethod
    def from_spec(cls, spec: "SessionWindow | dict | str | None") -> Optional["SessionWindow"]:
        if spec is None:
            return None
        if isinstance(spec, cls):
            return spec
        if isinstance(spec, str):
            name = spec.strip().upper()
            if not name:
                return None
            window = SESSION_PRESETS.get(name)
            if window is None:
                raise ValueError(f"Unknown session preset '{spec}'")
            start, end = window
            return cls(label=name, start_minute=_parse_time_string(start), end_minute=_parse_time_string(end))
        if isinstance(spec, dict):
            if {"start_minute", "end_minute"}.issubset(spec):
                return cls(
                    label=str(spec.get("label")) if spec.get("label") else None,
                    start_minute=int(spec["start_minute"]),
                    end_minute=int(spec["end_minute"]),
                )
            start_raw = spec.get("start")
            end_raw = spec.get("end")
            if not (start_raw and end_raw):
                raise ValueError("Session mapping requires 'start' and 'end' keys")
            return cls(
                label=str(spec.get("label")) if spec.get("label") else None,
                start_minute=_parse_time_string(str(start_raw)),
                end_minute=_parse_time_string(str(end_raw)),
            )
        raise TypeError(f"Unsupported session specification type: {type(spec)!r}")


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
    session: Optional[SessionWindow | dict | str] = None
    atr_n: Optional[int] = None
    atr_k: Optional[float] = None
    tp_mode: str = "OFF"  # "ATR", "BPS", or "OFF"
    sl_bps: float = 0.0
    tp_bps: float = 0.0

    def __post_init__(self) -> None:
        self.session = SessionWindow.from_spec(self.session)
        # Normalize ATR parameters for backwards compatibility
        if self.atr_n is None:
            self.atr_n = int(self.atr_window)
        else:
            self.atr_window = int(self.atr_n)
        if self.atr_k is None:
            self.atr_k = float(self.tp_multiplier)
        else:
            self.atr_k = float(self.atr_k)
        mode = (self.tp_mode or "OFF").upper()
        if mode not in {"ATR", "BPS", "OFF"}:
            raise ValueError(f"Unsupported tp_mode '{self.tp_mode}'")
        self.tp_mode = mode
        if mode in {"ATR", "BPS"}:
            self.use_atr_targets = True
        self.sl_bps = float(self.sl_bps)
        self.tp_bps = float(self.tp_bps)

    def as_dict(self) -> Dict[str, object]:
        session_dict: Optional[Dict[str, object]] = None
        if isinstance(self.session, SessionWindow):
            session_dict = self.session.to_dict()
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
            "session": session_dict,
            "atr_n": self.atr_n,
            "atr_k": self.atr_k,
            "tp_mode": self.tp_mode,
            "sl_bps": self.sl_bps,
            "tp_bps": self.tp_bps,
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
        total_gain = float(wins.sum())
        total_loss = float(-losses.sum())
        if total_loss > 1e-9:
            profit_factor = total_gain / total_loss
        elif total_gain > 1e-9:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0
        downside = losses
        if downside.size > 0:
            downside_std = float(np.sqrt(np.mean(np.square(downside))))
            sortino = (avg_return / downside_std) * math.sqrt(len(returns)) if downside_std > 1e-9 else 0.0
        else:
            sortino = 0.0
        return {
            "trade_count": int(len(self.trades)),
            "avg_return_bps": float(avg_return * 10_000.0),
            "win_rate": float((returns > 0).mean()),
            "payoff": float(payoff),
            "sharpe": float(sharpe),
            "calmar": float(calmar),
            "sortino": float(sortino),
            "profit_factor": float(profit_factor),
            "max_drawdown_pct": float(max_dd * 100.0),
            "avg_time_in_trade_min": float(avg_minutes),
            "alpha_vs_baseline_bps": float((avg_return - baseline_mean) * 10_000.0),
            "bootstrap_p_value": float(bootstrap_p) if bootstrap_p is not None else None,
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

    # ------------------------------------------------------------------
    # Export helpers

    def trade_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(
                columns=[
                    "instrument",
                    "entry_ts",
                    "exit_ts",
                    "direction",
                    "size",
                    "entry_price",
                    "exit_price",
                    "pnl_pct",
                    "pnl_bps",
                    "bars_held",
                    "hazard_on_exit",
                    "reason",
                    "entry_signal_index",
                    "exit_signal_index",
                ]
            )
        records = []
        for trade in self.trades:
            records.append(
                {
                    "instrument": trade.instrument,
                    "entry_ts": trade.entry_ts,
                    "exit_ts": trade.exit_ts,
                    "direction": "long" if trade.direction > 0 else "short",
                    "size": float(trade.size),
                    "entry_price": float(trade.entry_price),
                    "exit_price": float(trade.exit_price),
                    "pnl_pct": float(trade.pnl_pct),
                    "pnl_bps": float(trade.pnl_bps),
                    "bars_held": int(trade.bars_held),
                    "hazard_on_exit": float(trade.hazard_on_exit),
                    "reason": trade.reason,
                    "entry_signal_index": int(trade.entry_signal_index),
                    "exit_signal_index": int(trade.exit_signal_index),
                }
            )
        df = pd.DataFrame.from_records(records)
        df.sort_values("entry_ts", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def equity_curve(self) -> pd.DataFrame:
        trades_df = self.trade_frame()
        if trades_df.empty:
            return pd.DataFrame(columns=["timestamp", "equity_pct", "equity_bps", "trade_index"])
        equity_pct = trades_df["pnl_pct"].cumsum()
        curve = pd.DataFrame(
            {
                "timestamp": trades_df["exit_ts"],
                "equity_pct": equity_pct,
                "equity_bps": equity_pct * 10_000.0,
                "trade_index": np.arange(1, len(trades_df) + 1),
            }
        )
        return curve

    def hazard_calibration(self, bins: int = 20) -> pd.DataFrame:
        if self.signals.empty:
            return pd.DataFrame(columns=["hazard_bin", "signal_count", "entry_count", "admission_rate"])
        signals = self.signals.reset_index(drop=True).copy()
        entry_indices = {trade.entry_signal_index for trade in self.trades}
        signals["is_entry"] = signals.index.isin(entry_indices)
        bins = max(1, bins)
        hazard_bins = np.linspace(0.0, 1.0, bins + 1)
        signals["hazard_bin"] = pd.cut(
            signals["lambda_hazard"], bins=hazard_bins, include_lowest=True, right=True
        )
        grouped = signals.groupby("hazard_bin", observed=False)
        stats = grouped.agg(
            signal_count=("lambda_hazard", "size"),
            entry_count=("is_entry", "sum"),
            hazard_mean=("lambda_hazard", "mean"),
        ).reset_index()
        stats["admission_rate"] = stats.apply(
            lambda row: float(row["entry_count"]) / row["signal_count"] if row["signal_count"] else np.nan,
            axis=1,
        )
        return stats

    def lead_time_minutes(self) -> pd.Series:
        if not self.trades:
            return pd.Series(dtype=float)
        minutes = pd.Series([trade.bars_held for trade in self.trades], name="bars_held", dtype=float)
        return minutes

    def hazard_scatter_frame(self) -> pd.DataFrame:
        if self.signals.empty:
            return pd.DataFrame(columns=["lambda_hazard", "repetition_count"])
        return self.signals[["lambda_hazard", "repetition_count"]].copy()


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
        self._session_window: Optional[SessionWindow] = (
            self.config.session if isinstance(self.config.session, SessionWindow) else None
        )
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
        if self._session_window and not self._session_window.contains(int(row["timestamp_ms"])):
            return False
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
        mode = self.config.tp_mode
        entry = float(row["price"])
        if mode == "BPS":
            basis = 10_000.0
            sl_bps = max(0.0, self.config.sl_bps)
            tp_bps = max(0.0, self.config.tp_bps)
            if sl_bps <= 0 and tp_bps <= 0:
                return entry * 0.99, entry * 1.01
            sl_offset = entry * (sl_bps / basis)
            tp_offset = entry * (tp_bps / basis)
            if direction > 0:
                sl = entry - sl_offset if sl_bps > 0 else entry * 0.99
                tp = entry + tp_offset if tp_bps > 0 else entry * 1.01
            else:
                sl = entry + sl_offset if sl_bps > 0 else entry * 1.01
                tp = entry - tp_offset if tp_bps > 0 else entry * 0.99
            return sl, tp

        atr = float(row.get("atr", 0.0))
        if atr <= 0:
            return entry * 0.99, entry * 1.01
        sl_mult = max(1e-4, self.config.sl_multiplier)
        tp_source = self.config.atr_k if mode == "ATR" else self.config.tp_multiplier
        tp_mult = max(1e-4, tp_source)
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
    instrument_overrides: Optional[Dict[str, Dict[str, object]]] = None,
    capture_details: bool = False,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Run a grid sweep of strategy parameters across instruments."""
    results: List[Dict[str, object]] = []
    details: List[Dict[str, object]] = []
    overrides = instrument_overrides or {}
    for params in parameter_grid(grid):
        all_trades: List[Trade] = []
        baseline_returns = []
        per_instrument: Dict[str, Dict[str, object]] = {}
        instrument_results: Dict[str, BacktestResult] = {}
        for inst, candles in candles_by_instrument.items():
            signals = signals_by_instrument[inst]
            cfg_dict = {**base_config.as_dict(), **params, **overrides.get(inst, {})}
            cfg = StrategyConfig(**cfg_dict)
            runner = EchoBacktester(candles=candles, signals=signals, instrument=inst, config=cfg)
            res = runner.run()
            all_trades.extend(res.trades)
            baseline_returns.append(res.baseline)
            per_instrument[inst] = res.summary()
            if capture_details:
                instrument_results[inst] = res

        merged_signals = pd.concat(baseline_returns) if baseline_returns else pd.Series(dtype=float)
        aggregate_cfg = StrategyConfig(**{**base_config.as_dict(), **params})
        bt = BacktestResult(trades=all_trades, baseline=merged_signals, signals=pd.DataFrame(), config=aggregate_cfg)
        summary = bt.summary()
        summary.update({"instrument_count": len(candles_by_instrument), "params": params})
        summary["bootstrap_p_value"] = BacktestResult._bootstrap_p_value(
            np.array([t.pnl_pct for t in all_trades]), iterations=bootstrap_iterations, seed=seed
        )
        summary["instrument_summaries"] = per_instrument
        results.append(summary)
        if capture_details:
            details.append({
                "params": params,
                "portfolio_result": bt,
                "instrument_results": instrument_results,
            })
    return results, details


def dump_results(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
