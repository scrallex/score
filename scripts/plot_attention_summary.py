#!/usr/bin/env python3
"""Generate consolidated attention diagnostics from aggregated metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt

METRICS_FILE = Path("results/analysis/attention_metrics.json")
OUTPUT_PATH = Path("docs/figures/attention_summary.png")

# Dataset groupings and human-friendly labels.
DATASETS: Sequence[Tuple[str, Sequence[Tuple[str, str]]]] = (
    (
        "HoVer",
        (
            ("hover_val_fever_base", "FEVER init"),
            ("hover_val_fever_adapt", "HoVer adapt"),
        ),
    ),
    (
        "SciFact",
        (
            ("scifact_val_fever_init", "FEVER init"),
            ("scifact_val_finetune", "SciFact ft"),
            ("scifact_val_curriculum", "Curriculum"),
        ),
    ),
    (
        "FEVER",
        (
            ("fever_val", "Baseline"),
            ("fever_val_curriculum", "Curriculum"),
        ),
    ),
)

# Feature key, axis label, and optional error key (for std bars).
FEATURES: Sequence[Tuple[str, str, str]] = (
    ("mean_probability", "Mean admit probability", "prob_std"),
    ("max_attention_mean", "Mean max attention", "max_attention_std"),
    ("attention_entropy_mean", "Attention entropy", "attention_entropy_std"),
)

BAR_WIDTH = 0.24
GROUP_GAP = 0.55


def load_metrics(path: Path) -> Dict[str, Dict[str, float]]:
    return json.loads(path.read_text())


def compute_positions() -> Tuple[List[List[float]], List[float]]:
    """Return bar positions per dataset and the centre ticks for xticks."""

    positions: List[List[float]] = []
    centres: List[float] = []
    cursor = 0.0
    for _, entries in DATASETS:
        local: List[float] = []
        for _ in entries:
            local.append(cursor)
            cursor += BAR_WIDTH
        positions.append(local)
        centres.append(sum(local) / len(local))
        cursor += GROUP_GAP
    return positions, centres


def plot_feature(
    ax: plt.Axes,
    metrics: Dict[str, Dict[str, float]],
    feature_key: str,
    error_key: str,
    colors: Sequence[str],
    positions: List[List[float]],
) -> None:
    for dataset_idx, (_, entries) in enumerate(DATASETS):
        for entry_idx, (metric_key, label) in enumerate(entries):
            entry = metrics.get(metric_key, {})
            value = entry.get(feature_key, float("nan"))
            if error_key:
                error = entry.get(error_key, 0.0)
            else:
                error = 0.0
            pos = positions[dataset_idx][entry_idx]
            ax.bar(
                pos,
                value,
                BAR_WIDTH * 0.95,
                color=colors[entry_idx % len(colors)],
                edgecolor="black",
                linewidth=0.35,
                label=label if dataset_idx == 0 else "_nolegend_",
                yerr=error if error else None,
                capsize=3,
            )


def main() -> None:
    metrics = load_metrics(METRICS_FILE)
    positions, centres = compute_positions()

    fig, axes = plt.subplots(len(FEATURES), 1, figsize=(9, 9), sharex=True)
    if hasattr(axes, "ravel"):
        axes = list(axes.ravel())
    elif not isinstance(axes, (list, tuple)):
        axes = [axes]

    palette = plt.get_cmap("tab10")
    colors = [palette(i) for i in range(5)]

    for axis_idx, (feature_key, label, error_key) in enumerate(FEATURES):
        ax = axes[axis_idx]
        plot_feature(ax, metrics, feature_key, error_key, colors, positions)
        ax.set_ylabel(label)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.axhline(0.0, color="black", linewidth=0.3)
        if axis_idx == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xticks(centres)
    axes[-1].set_xticklabels([group for group, _ in DATASETS])
    axes[-1].set_xlabel("Dataset")

    xmin = positions[0][0] - BAR_WIDTH
    xmax = positions[-1][-1] + BAR_WIDTH
    for ax in axes:
        ax.set_xlim(xmin, xmax)

    fig.suptitle("Attention diagnostics across datasets")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=220)
    plt.close(fig)


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
