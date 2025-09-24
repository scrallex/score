#!/usr/bin/env python3
"""Run STM echo backtests and parameter sweeps."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sep_common.echo_strategy_profile import load_strategy_profile

from stm_backtest import build_signals, load_candles_csv, preprocess_candles
from stm_backtest.backtester import StrategyConfig, dump_results, run_sweep
from stm_backtest.reporting import export_run_report, render_param_key


KEY_MAP = {
    "coherence_min": "min_coherence",
    "stability_min": "min_stability",
    "entropy_max": "max_entropy",
    "hazard_max": "max_hazard",
    "atr_n": "atr_n",
    "atr_k": "atr_k",
    "tp_mode": "tp_mode",
    "sl_bps": "sl_bps",
    "tp_bps": "tp_bps",
    "session": "session",
}


def normalize_params(raw: dict | None) -> dict:
    if not raw:
        return {}
    normalized: Dict[str, object] = {}
    for key, value in raw.items():
        target = KEY_MAP.get(key, key)
        normalized[target] = value
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STM echo backtesting pipeline")
    parser.add_argument("--config", required=True, help="YAML configuration for the sweep")
    parser.add_argument("--output", help="Optional override for summary JSON output")
    parser.add_argument("--verbose", action="store_true", help="Print intermediate progress")
    parser.add_argument("--report", help="Optional directory to write per-run artifacts")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_output_paths(cfg: dict, override: str | None) -> Dict[str, Path]:
    out_cfg = cfg.get("output", {})
    base = Path(out_cfg.get("dir", "score/output/echo_backtest"))
    base.mkdir(parents=True, exist_ok=True)
    summary_json = Path(override) if override else base / out_cfg.get("summary_json", "sweep.json")
    summary_csv = base / out_cfg.get("summary_csv", "sweep.csv")
    signals_dir = Path(out_cfg.get("signals_dir", base / "signals"))
    return {"summary_json": summary_json, "summary_csv": summary_csv, "signals_dir": signals_dir}


def build_or_load_signals(
    *,
    candles: pd.DataFrame,
    instrument: str,
    cfg: dict,
    cache_dir: Path,
    verbose: bool = False,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    sig_prec = int(cfg.get("signature_precision", 2))
    lookback = float(cfg.get("lookback_minutes", 60.0))
    max_signals = cfg.get("max_signals")
    step_divisor = int(cfg.get("step_divisor", 32))
    cache_name = f"{instrument}_sig{sig_prec}_lookback{int(lookback)}_step{step_divisor}.csv.gz"
    cache_path = cache_dir / cache_name
    if cache_path.exists():
        if verbose:
            print(f"[signals] loading {cache_path}")
        signals = pd.read_csv(cache_path, parse_dates=["timestamp"])
        for col in ("timestamp_ms", "timestamp_ns", "repetition_count", "repetition_first_seen_ms", "window_index"):
            if col in signals:
                signals[col] = pd.to_numeric(signals[col], errors="coerce").fillna(0).astype("int64")
        return signals

    if verbose:
        print(f"[signals] building manifold for {instrument}")
    signals = build_signals(
        candles,
        instrument=instrument,
        signature_precision=sig_prec,
        lookback_minutes=lookback,
        max_signals=max_signals,
        step_divisor=step_divisor,
    )
    signals.to_csv(cache_path, index=False)
    return signals


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    raw_profile_path = (
        cfg.get("strategy", {}).get("profile_path")
        or (cfg.get("strategy_profile") or {}).get("path")
        or cfg.get("profile_path")
        or (cfg.get("profile") or {}).get("path")
        or os.getenv("ECHO_STRATEGY_PROFILE")
    )
    if raw_profile_path:
        candidate = Path(raw_profile_path)
        profile_path = candidate if candidate.is_absolute() else (cfg_path.parent / candidate).resolve()
    else:
        profile_path = REPO_ROOT / "config" / "echo_strategy.yaml"
    try:
        strategy_profile = load_strategy_profile(profile_path)
    except Exception as exc:
        strategy_profile = None
        if args.verbose:
            print(f"[profile] fallback to config overrides ({exc})")

    outputs = resolve_output_paths(cfg, args.output)

    data_cfg = cfg.get("data", {})
    data_root = Path(data_cfg.get("root", "."))
    instruments_cfg = data_cfg.get("instruments", [])
    if not instruments_cfg:
        raise ValueError("Config must provide data.instruments")
    enforce_m1 = bool(data_cfg.get("enforce_m1", True))

    signal_cfg = cfg.get("signals", {})
    cache_root = Path(signal_cfg.get("cache_dir", outputs["signals_dir"]))

    candles_by_inst: Dict[str, pd.DataFrame] = {}
    signals_by_inst: Dict[str, pd.DataFrame] = {}
    overrides_by_inst: Dict[str, Dict[str, object]] = {}

    for inst_cfg in instruments_cfg:
        symbol = inst_cfg.get("symbol")
        if not symbol:
            raise ValueError("Instrument entry missing 'symbol'")
        rel_path = inst_cfg.get("file")
        if not rel_path:
            raise ValueError(f"Instrument {symbol} missing 'file'")
        csv_path = data_root / rel_path
        if args.verbose:
            print(f"[data] loading {symbol} from {csv_path}")
        timestamp_col = inst_cfg.get("timestamp_col")
        tz = inst_cfg.get("tz", "UTC")
        raw = load_candles_csv(csv_path, timestamp_col=timestamp_col, tz=tz)
        candles = preprocess_candles(raw, enforce_m1=enforce_m1)
        symbol_key = str(symbol).upper()
        candles_by_inst[symbol_key] = candles
        signals = build_or_load_signals(
            candles=candles,
            instrument=symbol_key,
            cfg=signal_cfg,
            cache_dir=cache_root,
            verbose=args.verbose,
        )
        signals_by_inst[symbol_key] = signals
        overrides_by_inst[symbol_key] = normalize_params(inst_cfg.get("overrides", {}))

    strategy_cfg = cfg.get("strategy", {})
    base_params = normalize_params(strategy_cfg.get("base", {}))
    grid_params = normalize_params(strategy_cfg.get("grid", cfg.get("grid", {})))

    if strategy_profile:
        global_cfg = strategy_profile.global_config
        guards = strategy_profile.guard_defaults()
        base_params.setdefault("min_repetitions", global_cfg.get("min_repetitions", 3))
        base_params.setdefault("max_hazard", global_cfg.get("hazard_max"))
        base_params.setdefault("exit_horizon", global_cfg.get("exit_horizon"))
        base_params.setdefault("hazard_exit_threshold", global_cfg.get("hazard_exit_threshold"))
        base_params.setdefault("min_coherence", guards.get("min_coherence"))
        base_params.setdefault("min_stability", guards.get("min_stability"))
        base_params.setdefault("max_entropy", guards.get("max_entropy"))

    if strategy_profile:
        for symbol, profile in strategy_profile.instruments.items():
            key = symbol.upper()
            overrides = overrides_by_inst.setdefault(key, {})
            thresholds = profile.guard_thresholds()
            if thresholds.get("min_coherence") is not None:
                overrides.setdefault("min_coherence", thresholds.get("min_coherence"))
            if thresholds.get("min_stability") is not None:
                overrides.setdefault("min_stability", thresholds.get("min_stability"))
            if thresholds.get("max_entropy") is not None:
                overrides.setdefault("max_entropy", thresholds.get("max_entropy"))
            if profile.hazard_max is not None:
                overrides.setdefault("max_hazard", profile.hazard_max)
            if profile.min_repetitions is not None:
                overrides.setdefault("min_repetitions", profile.min_repetitions)
            if profile.session:
                overrides.setdefault("session", profile.session.to_range_spec())
            if profile.exit:
                for ekey, value in profile.exit.items():
                    if value is not None:
                        overrides.setdefault(ekey, value)
    base_config = StrategyConfig(**{**StrategyConfig().as_dict(), **base_params})

    bootstrap_cfg = cfg.get("bootstrap", {})
    iterations = int(bootstrap_cfg.get("iterations", 500))
    seed = bootstrap_cfg.get("seed")

    report_cfg = cfg.get("report", {})
    report_dir = args.report or report_cfg.get("outdir")
    report_path: Path | None = Path(report_dir) if report_dir else None
    make_plots = bool(report_cfg.get("make_plots", True)) if report_cfg else True

    if args.verbose:
        print("[sweep] running parameter grid")
    results, run_details = run_sweep(
        candles_by_instrument=candles_by_inst,
        signals_by_instrument=signals_by_inst,
        base_config=base_config,
        grid=grid_params,
        bootstrap_iterations=iterations,
        seed=seed,
        instrument_overrides=overrides_by_inst,
        capture_details=bool(report_path),
    )

    payload = {
        "config": {
            "base": base_config.as_dict(),
            "grid": grid_params,
            "bootstrap_iterations": iterations,
            "seed": seed,
            "instrument_overrides": overrides_by_inst,
        },
        "results": results,
    }

    dump_results(payload, outputs["summary_json"])
    df = pd.DataFrame(results)
    df.to_csv(outputs["summary_csv"], index=False)

    if report_path:
        if args.verbose:
            print(f"[report] writing artifacts to {report_path}")
        for summary, detail in zip(results, run_details, strict=False):
            run_dir = report_path / render_param_key(summary.get("params", {}))
            export_run_report(
                run_dir,
                summary,
                detail["portfolio_result"],
                detail["instrument_results"],
                make_plots=make_plots,
            )

    if args.verbose:
        print(f"[done] wrote {outputs['summary_json']} and {outputs['summary_csv']}")


if __name__ == "__main__":
    main()
