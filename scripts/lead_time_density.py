#!/usr/bin/env python3
"""Compute foreground density in lead-time bins."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Foreground density vs lead time")
    parser.add_argument("state", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--start", type=str, required=True, help="Window start timestamp (ISO-8601)")
    parser.add_argument("--stop", type=str, required=True, help="Window stop timestamp (ISO-8601)")
    parser.add_argument("--onset", type=str, required=True, help="Event onset timestamp (ISO-8601)")
    parser.add_argument("--bin", type=int, default=5, help="Bin size in minutes")
    parser.add_argument("--lookback", type=int, default=20, help="Minutes to look back from onset")
    return parser.parse_args()


def load_state(path: Path) -> Tuple[np.ndarray, int]:
    state = json.loads(path.read_text(encoding="utf-8"))
    metrics = np.array(
        [
            [sig["metrics"].get("coherence", 0.0), sig["metrics"].get("entropy", 1.0), sig["metrics"].get("stability", 0.0)]
            for sig in state.get("signals", [])
        ],
        dtype=float,
    )
    return metrics, len(state.get("signals", []))


def load_thresholds(config_path: Path) -> Tuple[float | None, float | None, float | None]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    fg = cfg["router"]["foreground"]
    return fg.get("min_coh"), fg.get("max_ent"), fg.get("min_stab")


def build_timeline(count: int, start: datetime, stop: datetime) -> np.ndarray:
    if count == 0:
        return np.array([])
    delta = (stop - start) / count
    return np.array([start + (i + 0.5) * delta for i in range(count)])


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


def summarise_bins(times: np.ndarray, fg_mask: np.ndarray, onset: datetime, bins: List[Tuple[int, int]]) -> List[dict]:
    results: List[dict] = []
    for lo, hi in bins:
        bin_start = onset + timedelta(minutes=lo)
        bin_end = onset + timedelta(minutes=hi)
        idx = (times >= bin_start) & (times < bin_end)
        total = int(idx.sum())
        fg = int((fg_mask & idx).sum())
        density = (fg / total) if total else 0.0
        results.append({"bin": f"{lo}..{hi} min", "windows": total, "foreground": fg, "density": round(density, 4)})
    return results


def main() -> None:
    args = parse_args()
    metrics, count = load_state(args.state)
    thresholds = load_thresholds(args.config)
    start = datetime.fromisoformat(args.start)
    stop = datetime.fromisoformat(args.stop)
    onset = datetime.fromisoformat(args.onset)

    times = build_timeline(count, start, stop)
    fg_mask = foreground_mask(metrics, thresholds)

    bins: List[Tuple[int, int]] = []
    lookback = abs(args.lookback)
    for offset in range(-lookback, 0, args.bin):
        bins.append((offset, min(offset + args.bin, 0)))

    summary = summarise_bins(times, fg_mask, onset, bins)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
