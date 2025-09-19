#!/usr/bin/env python3
"""Compute foreground density in lead-time bins and optional plot."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class LeadBin:
    label: str
    window_count: int
    foreground_count: int
    density: float
    midpoint_minutes: float

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "window_count": self.window_count,
            "foreground_count": self.foreground_count,
            "density": round(self.density, 4),
        }


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


def summarise_bins(
    times: np.ndarray,
    fg_mask: np.ndarray,
    onset: datetime,
    bins: Iterable[Tuple[int, int]],
) -> List[LeadBin]:
    results: List[LeadBin] = []
    for lo, hi in bins:
        bin_start = onset + timedelta(minutes=lo)
        bin_end = onset + timedelta(minutes=hi)
        idx = (times >= bin_start) & (times < bin_end)
        total = int(idx.sum())
        fg = int((fg_mask & idx).sum())
        density = (fg / total) if total else 0.0
        midpoint = (lo + hi) / 2.0
        results.append(
            LeadBin(
                label=f"{lo}..{hi} min",
                window_count=total,
                foreground_count=fg,
                density=density,
                midpoint_minutes=midpoint,
            )
        )
    return results


def compute_lead_summary(
    state_path: Path,
    config_path: Path,
    start: datetime,
    stop: datetime,
    onset: datetime,
    bin_minutes: int = 5,
    lookback_minutes: int = 20,
) -> dict:
    metrics, count = load_state(state_path)
    thresholds = load_thresholds(config_path)
    times = build_timeline(count, start, stop)
    fg_mask = foreground_mask(metrics, thresholds)

    bins: List[Tuple[int, int]] = []
    lookback_minutes = abs(lookback_minutes)
    for offset in range(-lookback_minutes, 0, bin_minutes):
        bins.append((offset, min(offset + bin_minutes, 0)))

    bin_summaries = summarise_bins(times, fg_mask, onset, bins)
    densities = [bin.density for bin in bin_summaries]
    monotonic = all(b >= a for a, b in zip(densities, densities[1:]))
    payload = {
        "bins": [bin.as_dict() for bin in bin_summaries],
        "monotonic_increase": monotonic,
    }
    return payload


def plot_lead_summary(summary: dict, output_path: Path) -> None:
    bins = summary.get("bins", [])
    if not bins:
        return
    x = [float(item["label"].split("..", 1)[0]) for item in bins]
    y = [item["density"] for item in summary["bins"]]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(x, y, marker="o")
    ax.set_xlabel("Minutes before onset")
    ax.set_ylabel("Foreground density")
    ax.set_title("Lead-time foreground density")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(min(x), 0)
    ax.set_ylim(0, max(max(y) * 1.1, 0.01))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Foreground density vs lead time")
    parser.add_argument("state", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--start", type=str, required=True, help="Window start timestamp (ISO-8601)")
    parser.add_argument("--stop", type=str, required=True, help="Window stop timestamp (ISO-8601)")
    parser.add_argument("--onset", type=str, required=True, help="Event onset timestamp (ISO-8601)")
    parser.add_argument("--bin", type=int, default=5, help="Bin size in minutes")
    parser.add_argument("--lookback", type=int, default=20, help="Minutes to look back from onset")
    parser.add_argument("--plot", type=Path, help="Optional path for PNG line plot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = datetime.fromisoformat(args.start)
    stop = datetime.fromisoformat(args.stop)
    onset = datetime.fromisoformat(args.onset)
    summary = compute_lead_summary(
        state_path=args.state,
        config_path=args.config,
        start=start,
        stop=stop,
        onset=onset,
        bin_minutes=args.bin,
        lookback_minutes=args.lookback,
    )
    if args.plot:
        plot_lead_summary(summary, args.plot)
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
