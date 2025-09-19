#!/usr/bin/env python3
"""Permutation test for final lead-time bin foreground density."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import numpy as np
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Permutation test for lead-time density")
    parser.add_argument("state", type=Path, help="State JSON path")
    parser.add_argument("config", type=Path, help="Router config JSON path")
    parser.add_argument("--start", required=True, help="Slice start timestamp (ISO-8601)")
    parser.add_argument("--stop", required=True, help="Slice stop timestamp (ISO-8601)")
    parser.add_argument("--onset", required=True, help="Onset timestamp (ISO-8601)")
    parser.add_argument("--lookback", type=int, default=5, help="Minutes before onset for the bin (default: 5)")
    parser.add_argument("--jitter", type=int, default=30, help="Uniform jitter window in minutes (default: ±30)")
    parser.add_argument("--iterations", type=int, default=1000, help="Permutation iterations (default: 1000)")
    parser.add_argument("--seed", type=int, help="Random seed")
    return parser.parse_args()


def foreground_mask(metrics: np.ndarray, thresholds: Tuple[float | None, float | None, float | None]) -> np.ndarray:
    min_coh, max_ent, min_stab = thresholds
    mask = np.ones(len(metrics), dtype=bool)
    if min_coh is not None:
        mask &= metrics[:, 0] >= float(min_coh)
    if max_ent is not None:
        mask &= metrics[:, 1] <= float(max_ent)
    if min_stab is not None:
        mask &= metrics[:, 2] >= float(min_stab)
    return mask


def final_bin_density(
    times: np.ndarray,
    fg_mask: np.ndarray,
    onset: datetime,
    lookback_minutes: int,
) -> float:
    lo = onset + timedelta(minutes=-lookback_minutes)
    hi = onset
    idx = (times >= lo) & (times < hi)
    total = int(idx.sum())
    if total == 0:
        return 0.0
    fg = int((fg_mask & idx).sum())
    return fg / total


def load_state(path: Path) -> Tuple[np.ndarray, int]:
    state = json.loads(path.read_text(encoding="utf-8"))
    metrics = np.array([
        [
            sig["metrics"].get("coherence", 0.0),
            sig["metrics"].get("entropy", 1.0),
            sig["metrics"].get("stability", 0.0),
        ]
        for sig in state.get("signals", [])
    ])
    return metrics, len(state.get("signals", []))


def load_thresholds(path: Path) -> Tuple[float | None, float | None, float | None]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    foreground = cfg["router"]["foreground"]
    return (
        foreground.get("min_coh"),
        foreground.get("max_ent"),
        foreground.get("min_stab"),
    )


def build_timeline(count: int, start: datetime, stop: datetime) -> np.ndarray:
    if count == 0:
        return np.array([])
    delta = (stop - start) / count
    return np.array([start + (i + 0.5) * delta for i in range(count)])


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    start = datetime.fromisoformat(args.start)
    stop = datetime.fromisoformat(args.stop)
    onset = datetime.fromisoformat(args.onset)

    metrics, count = load_state(args.state)
    thresholds = load_thresholds(args.config)
    times = build_timeline(count, start, stop)
    fg_mask = foreground_mask(metrics, thresholds)

    observed = final_bin_density(times, fg_mask, onset, args.lookback)

    jit = args.jitter
    null_samples = []
    for _ in range(args.iterations):
        delta = timedelta(minutes=random.uniform(-jit, jit))
        null_samples.append(final_bin_density(times, fg_mask, onset + delta, args.lookback))

    greater_equal = sum(1 for value in null_samples if value >= observed)
    p_value = (greater_equal + 1) / (args.iterations + 1)

    result = {
        "observed_density": observed,
        "iterations": args.iterations,
        "jitter_minutes": jit,
        "lookback_minutes": args.lookback,
        "p_value": p_value,
        "null_mean": float(np.mean(null_samples)) if null_samples else 0.0,
        "null_std": float(np.std(null_samples)) if null_samples else 0.0,
    }
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
