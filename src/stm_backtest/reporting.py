"""Reporting helpers for STM backtests."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtester import BacktestResult


def render_param_key(params: Dict[str, object]) -> str:
    """Convert a parameter dict into a filesystem-friendly key."""
    parts = []
    for key in sorted(params):
        value = params[key]
        if isinstance(value, float):
            value = f"{value:.4g}".replace("-", "neg").replace(".", "p")
        parts.append(f"{key}-{value}")
    return "__".join(parts)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _coerce_json(value):
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, (np.integer,)):
        value = int(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (pd.Series, pd.DataFrame)):
        return value.to_dict(orient="records")
    if isinstance(value, dict):
        return {k: _coerce_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(_coerce_json(payload), indent=2), encoding="utf-8")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        frame.to_csv(path, index=False)
    else:
        frame.to_csv(path, index=False)


def plot_equity_curve(equity: pd.DataFrame, path: Path, *, title: str) -> None:
    if equity.empty:
        return
    plt.figure(figsize=(10, 4))
    plt.plot(equity["timestamp"], equity["equity_bps"], color="#005f73")
    plt.title(title)
    plt.xlabel("Exit Time")
    plt.ylabel("Equity (bps)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_hazard_calibration(calibration: pd.DataFrame, path: Path) -> None:
    if calibration.empty:
        return
    plt.figure(figsize=(8, 4))
    plt.plot(calibration["hazard_mean"], calibration["admission_rate"], marker="o", color="#0a9396")
    plt.title("Admission Rate vs Hazard")
    plt.xlabel("Hazard λ (mean per bin)")
    plt.ylabel("Admission Rate")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_hazard_scatter(scatter: pd.DataFrame, path: Path) -> None:
    if scatter.empty:
        return
    plt.figure(figsize=(8, 4))
    plt.scatter(scatter["lambda_hazard"], scatter["repetition_count"], s=12, alpha=0.4, color="#94d2bd")
    plt.title("Echo Count vs Hazard λ")
    plt.xlabel("Hazard λ")
    plt.ylabel("Repetition Count")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_lead_time_histogram(lead_series: pd.Series, path: Path) -> None:
    if lead_series.empty:
        return
    plt.figure(figsize=(8, 4))
    plt.hist(lead_series, bins=20, color="#ee9b00", alpha=0.75)
    plt.title("Lead Time Histogram (bars)")
    plt.xlabel("Bars Held")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def export_run_report(
    base_dir: Path,
    summary: Dict[str, object],
    portfolio_result: BacktestResult,
    instrument_results: Dict[str, BacktestResult],
    *,
    make_plots: bool = True,
) -> None:
    ensure_dir(base_dir)

    # Metrics JSON
    metrics_path = base_dir / "metrics.json"
    metrics_payload = dict(summary)
    metrics_payload["params"] = summary.get("params", {})
    metrics_payload["instrument_summaries"] = summary.get("instrument_summaries", {})
    _write_json(metrics_path, metrics_payload)

    # Portfolio exports
    portfolio_trades = portfolio_result.trade_frame()
    portfolio_trades_path = base_dir / "trades.csv"
    _write_csv(portfolio_trades_path, portfolio_trades)

    portfolio_equity = portfolio_result.equity_curve()
    equity_csv_path = base_dir / "equity_curve.csv"
    _write_csv(equity_csv_path, portfolio_equity)
    if make_plots:
        plot_equity_curve(portfolio_equity, base_dir / "equity_curve.png", title="Portfolio Equity Curve")
        plot_lead_time_histogram(portfolio_result.lead_time_minutes(), base_dir / "lead_time_hist.png")

    # Per-instrument exports
    for instrument, result in instrument_results.items():
        inst_dir = base_dir / f"instrument_{instrument.lower()}"
        ensure_dir(inst_dir)
        _write_csv(inst_dir / "trades.csv", result.trade_frame())
        inst_equity = result.equity_curve()
        _write_csv(inst_dir / "equity_curve.csv", inst_equity)
        if make_plots:
            plot_equity_curve(inst_equity, inst_dir / "equity_curve.png", title=f"{instrument} Equity")
            plot_hazard_calibration(result.hazard_calibration(), inst_dir / "hazard_calibration.png")
            plot_hazard_scatter(result.hazard_scatter_frame(), inst_dir / "hazard_scatter.png")
            plot_lead_time_histogram(result.lead_time_minutes(), inst_dir / "lead_time_hist.png")
