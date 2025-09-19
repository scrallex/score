#!/usr/bin/env python3
"""Plot MMS structural overlays for a given state window."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8-darkgrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Bx/Bz with structural overlays")
    parser.add_argument("--bits", type=Path, required=True, help="Path to bit-feature CSV")
    parser.add_argument("--numeric", type=Path, required=True, help="Path to numeric FGM CSV")
    parser.add_argument("--state", type=Path, required=True, help="Path to STM state JSON")
    parser.add_argument("--config", type=Path, required=True, help="Router config JSON with coverage data")
    parser.add_argument("--start", type=str, required=True, help="Start timestamp (ISO-8601)")
    parser.add_argument("--stop", type=str, required=True, help="Stop timestamp (ISO-8601)")
    parser.add_argument("--zoom-minutes", type=int, default=10, help="Window size for zoomed plot")
    parser.add_argument("--zoom-end", type=str, default=None, help="End timestamp for zoomed view (ISO-8601)")
    parser.add_argument("--out-prefix", type=Path, required=True, help="Prefix for generated PNG files")
    return parser.parse_args()


def load_series(bit_path: Path, numeric_path: Path, start: datetime, stop: datetime) -> pd.DataFrame:
    bits = pd.read_csv(bit_path, parse_dates=["time"])
    nums = pd.read_csv(numeric_path, parse_dates=["time"])
    mask = (bits["time"] >= start) & (bits["time"] <= stop)
    bits = bits.loc[mask].copy()
    mask_num = (nums["time"] >= start) & (nums["time"] <= stop)
    nums = nums.loc[mask_num].copy()
    for col in bits.columns:
        if col == "time":
            continue
        bits[col] = bits[col].fillna("").astype(str).ne("")
    merged = nums.merge(
        bits[[
            "time",
            "mms1_fgm_b_gse_srvy_l2_x__RANGEEXP",
            "mms1_fgm_b_gse_srvy_l2_z__ACCEL",
        ]],
        on="time",
        how="left",
    ).fillna(False)
    merged.rename(
        columns={
            "mms1_fgm_b_gse_srvy_l2_x__RANGEEXP": "x_rangeexp",
            "mms1_fgm_b_gse_srvy_l2_z__ACCEL": "z_accel",
        },
        inplace=True,
    )
    return merged


def load_foreground(state_path: Path, config_path: Path) -> tuple[np.ndarray, np.ndarray]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    thresholds = cfg["router"]["foreground"]
    min_coh = thresholds.get("min_coh")
    max_ent = thresholds.get("max_ent")
    min_stab = thresholds.get("min_stab")
    metrics = np.array(
        [
            [sig["metrics"].get("coherence", 0.0), sig["metrics"].get("entropy", 1.0), sig["metrics"].get("stability", 0.0)]
            for sig in state.get("signals", [])
        ],
        dtype=float,
    )
    fg_mask = np.ones(len(metrics), dtype=bool)
    if min_coh is not None:
        fg_mask &= metrics[:, 0] >= float(min_coh)
    if max_ent is not None:
        fg_mask &= metrics[:, 1] <= float(max_ent)
    if min_stab is not None:
        fg_mask &= metrics[:, 2] >= float(min_stab)
    return metrics, fg_mask


def foreground_edges(
    window_mask: np.ndarray,
    fg_mask: np.ndarray,
    full_start: float,
    full_stop: float,
) -> tuple[np.ndarray, np.ndarray]:
    total = len(fg_mask)
    if total == 0:
        return np.linspace(full_start, full_stop, 2), np.zeros(1)
    start_frac = np.clip((window_mask[0] - full_start) / (full_stop - full_start), 0.0, 1.0)
    stop_frac = np.clip((window_mask[1] - full_start) / (full_stop - full_start), 0.0, 1.0)
    start_idx = int(np.floor(start_frac * total))
    stop_idx = max(start_idx + 1, int(np.ceil(stop_frac * total)))
    edges = np.linspace(window_mask[0], window_mask[1], stop_idx - start_idx + 1)
    values = fg_mask[start_idx:stop_idx]
    if values.size == 0:
        edges = np.linspace(window_mask[0], window_mask[1], 2)
        values = np.zeros(1)
    return edges, values


def plot_window(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    edge_span: tuple[float, float],
    fg_mask: np.ndarray,
    full_start: float,
    full_stop: float,
) -> None:
    times = df["time"].to_numpy()
    bx = df["mms1_fgm_b_gse_srvy_l2_x"].astype(float).to_numpy()
    bz = df["mms1_fgm_b_gse_srvy_l2_z"].astype(float).to_numpy()
    rangeexp = df["x_rangeexp"].to_numpy(dtype=bool)
    accel = df["z_accel"].to_numpy(dtype=bool)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 3, 0.6]})

    ax_bx, ax_bz, ax_fg = axes
    ax_bx.plot(times, bx, "-", color="#1f77b4", lw=1.2, label="Bx")
    ax_bx.fill_between(times, bx, where=rangeexp, color="#ff7f0e", alpha=0.2, label="__rangeexp")
    ax_bx.set_ylabel("Bx [nT]")
    ax_bx.legend(loc="upper right")

    ax_bz.plot(times, bz, "-", color="#2ca02c", lw=1.2, label="Bz")
    ax_bz.fill_between(times, bz, where=accel, color="#d62728", alpha=0.2, label="__accel")
    ax_bz.set_ylabel("Bz [nT]")
    ax_bz.legend(loc="upper right")

    edges, values = foreground_edges(edge_span, fg_mask, full_start, full_stop)
    ax_fg.pcolormesh(edges, [0, 1], values.astype(float)[np.newaxis, :], cmap="inferno", vmin=0, vmax=1, shading="flat")
    ax_fg.set_yticks([])
    ax_fg.set_ylabel("Foreground")
    ax_fg.set_xlabel("UTC")

    formatter = mdates.DateFormatter("%H:%M")
    ax_fg.xaxis.set_major_formatter(formatter)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    start = datetime.fromisoformat(args.start)
    stop = datetime.fromisoformat(args.stop)
    merged = load_series(args.bits, args.numeric, start, stop)
    metrics, fg_mask = load_foreground(args.state, args.config)

    full_start_num = mdates.date2num(start)
    full_stop_num = mdates.date2num(stop)

    plot_window(
        merged,
        args.out_prefix.with_name(args.out_prefix.name + "_overview.png"),
        title=f"{start:%Y-%m-%d %H:%M} to {stop:%H:%M} UTC",
        edge_span=(full_start_num, full_stop_num),
        fg_mask=fg_mask,
        full_start=full_start_num,
        full_stop=full_stop_num,
    )

    zoom_end = datetime.fromisoformat(args.zoom_end) if args.zoom_end else stop
    zoom_start = zoom_end - timedelta(minutes=args.zoom_minutes)
    zoom_slice = merged[(merged["time"] >= zoom_start) & (merged["time"] <= zoom_end)].copy()
    zoom_start_num = mdates.date2num(zoom_start)
    zoom_end_num = mdates.date2num(zoom_end)
    plot_window(
        zoom_slice,
        args.out_prefix.with_name(args.out_prefix.name + "_zoom.png"),
        title=f"Zoom {zoom_start:%H:%M}–{zoom_end:%H:%M} UTC",
        edge_span=(zoom_start_num, zoom_end_num),
        fg_mask=fg_mask,
        full_start=full_start_num,
        full_stop=full_stop_num,
    )


if __name__ == "__main__":
    main()
