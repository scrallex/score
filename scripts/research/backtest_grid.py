#!/usr/bin/env python3
"""Run a grid of STM backtest simulations with partial output snapshots."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import orjson
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stm_backtest import load_candles_csv, preprocess_candles
from stm_backtest.backtester import StrategyConfig

from scripts.research.simulator.backtest_simulator import (
    BacktestSimulator,
    GuardThresholds,
    InstrumentPayload,
)


@dataclass
class InstrumentConfig:
    symbol: str
    candles_path: Path
    gates_path: Optional[Path]
    guards: GuardThresholds
    strategy_overrides: Dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML or JSON config describing the grid")
    parser.add_argument("--output-dir", help="Override output directory from config")
    parser.add_argument("--latest-name", default="latest", help="Base name for partial/final JSON artefacts")
    parser.add_argument("--verbose", action="store_true", help="Print progress to stderr")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    return json.loads(text)


def resolve_path(base: Path, value: str | None) -> Optional[Path]:
    if not value:
        return None
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def build_strategy(raw: dict | None) -> StrategyConfig:
    allowed = StrategyConfig.__dataclass_fields__.keys()
    params: Dict[str, object] = {}
    if raw:
        for key, value in raw.items():
            if key not in allowed:
                raise ValueError(f"Unknown strategy key '{key}'")
            params[key] = value
    return StrategyConfig(**params)


def build_guard_defaults(strategy: StrategyConfig) -> GuardThresholds:
    return GuardThresholds(
        min_repetitions=strategy.min_repetitions,
        max_hazard=strategy.max_hazard,
        min_coherence=strategy.min_coherence,
        min_stability=strategy.min_stability,
        max_entropy=strategy.max_entropy,
    )


def guard_from_config(raw: dict | None, defaults: GuardThresholds) -> GuardThresholds:
    if not raw:
        return defaults
    data = {
        "min_repetitions": raw.get("min_repetitions", defaults.min_repetitions),
        "max_hazard": raw.get("max_hazard", defaults.max_hazard),
        "min_coherence": raw.get("min_coherence", defaults.min_coherence),
        "min_stability": raw.get("min_stability", defaults.min_stability),
        "max_entropy": raw.get("max_entropy", defaults.max_entropy),
    }
    return GuardThresholds(**data)


def load_signals(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        sep = "," if suffix == ".csv" else "\t"
        return pd.read_csv(path, sep=sep)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported gate format: {path.suffix}")


def normalise_signals(frame: pd.DataFrame, instrument: str, candles: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    if df.empty:
        return df
    if "instrument" not in df.columns:
        df["instrument"] = instrument
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df[df["timestamp"].notna()].copy()
    if "timestamp_ms" not in df.columns and "timestamp" in df.columns:
        df["timestamp_ms"] = (df["timestamp"].view("int64") // 1_000_000).astype("int64")
    if "timestamp" not in df.columns and "timestamp_ms" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    if "timestamp_ns" not in df.columns and "timestamp_ms" in df.columns:
        df["timestamp_ns"] = df["timestamp_ms"].astype("int64") * 1_000_000
    if "repetition_first_seen_ms" not in df.columns and "timestamp_ms" in df.columns:
        df["repetition_first_seen_ms"] = df["timestamp_ms"]
    if "window_index" not in df.columns:
        df["window_index"] = range(len(df))
    if "price" not in df.columns and not candles.empty:
        price_map = candles.set_index("timestamp")["close"]
        df["price"] = df["timestamp"].map(price_map)
        df["price"].fillna(method="ffill", inplace=True)
        df["price"].fillna(method="bfill", inplace=True)
    required = {
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
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Derived signals missing columns: {', '.join(sorted(missing))}")
    df.sort_values("timestamp_ms", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def iter_instruments(cfg: dict, base_dir: Path, defaults: GuardThresholds) -> Iterable[InstrumentConfig]:
    items = cfg.get("instruments") or []
    if not items:
        raise ValueError("Config must define at least one instrument")
    for entry in items:
        symbol = entry.get("symbol")
        if not symbol:
            raise ValueError("Instrument entry missing 'symbol'")
        candles_rel = entry.get("candles") or entry.get("file")
        if not candles_rel:
            raise ValueError(f"Instrument {symbol} missing 'candles' path")
        candles_path = resolve_path(base_dir, candles_rel)
        if candles_path is None or not candles_path.exists():
            raise FileNotFoundError(f"Candles file not found for {symbol}: {candles_rel}")
        gates_path = resolve_path(base_dir, entry.get("gates"))
        guards = guard_from_config(entry.get("guards"), defaults)
        overrides = entry.get("overrides") or {}
        yield InstrumentConfig(
            symbol=str(symbol),
            candles_path=candles_path,
            gates_path=gates_path,
            guards=guards,
            strategy_overrides=overrides,
        )


def write_partial(path: Path, payload: dict) -> None:
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    base_dir = cfg_path.parent

    strategy_cfg = cfg.get("strategy", {})
    base_strategy = build_strategy(strategy_cfg.get("base"))
    builder_kwargs = strategy_cfg.get("builder", {})
    min_trades = int(cfg.get("min_trades", cfg.get("min_trades_required", 10)))

    defaults = build_guard_defaults(base_strategy)

    output_root = Path(args.output_dir) if args.output_dir else Path(cfg.get("output_dir", "output/backtests"))
    output_root.mkdir(parents=True, exist_ok=True)
    partial_path = output_root / f"{args.latest_name}.partial.json"
    final_path = output_root / f"{args.latest_name}.json"

    simulator = BacktestSimulator(
        base_strategy=base_strategy,
        builder_kwargs=builder_kwargs,
        min_trades=min_trades,
    )

    grid_entries = cfg.get("grid") or [{}]
    runs: List[Dict[str, object]] = []
    status = "running"

    for entry in grid_entries:
        run_name = entry.get("name") or f"run_{len(runs) + 1}"
        overrides = entry.get("overrides") or {}
        instrument_results: List[Dict[str, object]] = []
        for inst_cfg in iter_instruments(cfg, base_dir, defaults):
            candles_raw = load_candles_csv(inst_cfg.candles_path)
            candles = preprocess_candles(candles_raw)
            gates = None
            if inst_cfg.gates_path is not None:
                if not inst_cfg.gates_path.exists():
                    raise FileNotFoundError(f"Gates file not found for {inst_cfg.symbol}: {inst_cfg.gates_path}")
                gates_frame = load_signals(inst_cfg.gates_path)
                gates = normalise_signals(gates_frame, inst_cfg.symbol, candles)
            merged_overrides = dict(inst_cfg.strategy_overrides)
            merged_overrides.update(overrides)
            payload = InstrumentPayload(
                instrument=inst_cfg.symbol,
                candles=candles,
                guards=inst_cfg.guards,
                gates=gates,
                strategy_overrides=merged_overrides,
            )
            try:
                result = simulator.simulate_instrument(payload)
            except Exception as exc:  # pragma: no cover - surface errors in output
                if args.verbose:
                    print(f"[error] {inst_cfg.symbol}: {exc}", file=sys.stderr)
                result = {
                    "instrument": inst_cfg.symbol,
                    "error": str(exc),
                    "metrics": {"qualified": False, "trade_count": 0},
                    "trades": [],
                    "equity_curve": [],
                    "gate_coverage": 0.0,
                    "source": "error",
                }
            instrument_results.append(result)
            snapshot = {
                "status": status,
                "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "runs": runs + [
                    {
                        "name": run_name,
                        "overrides": overrides,
                        "instruments": instrument_results,
                    }
                ],
            }
            write_partial(partial_path, snapshot)

        runs.append(
            {
                "name": run_name,
                "overrides": overrides,
                "instruments": instrument_results,
            }
        )
        snapshot = {
            "status": status,
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "runs": runs,
        }
        write_partial(partial_path, snapshot)

    status = "completed"
    final_payload = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "runs": runs,
    }
    write_partial(partial_path, final_payload)
    os.replace(partial_path, final_path)
    if args.verbose:
        print(f"[done] wrote {final_path}")


if __name__ == "__main__":
    main()
