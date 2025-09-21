#!/usr/bin/env python3
"""Plot guardrail sweep overlay showing lead vs permutation significance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_summary(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    rows = [row for row in rows if row.get("target_guardrail") is not None]
    rows.sort(key=lambda item: item["target_guardrail"])
    return rows


def build_plot(
    rows: list[dict],
    *,
    highlight: float | None,
    output_path: Path,
    title: str | None = None,
) -> None:
    if not rows:
        raise ValueError("No rows found in sweep summary")

    targets = np.array([row["target_guardrail"] for row in rows], dtype=float) * 100.0
    lead_mean = np.array([row.get("lead_mean", 0.0) or 0.0 for row in rows], dtype=float)
    p_min = np.array([row.get("p_value_min", 0.0) or 0.0 for row in rows], dtype=float)
    coverage = np.array([row.get("actual_coverage_pct", 0.0) or 0.0 for row in rows], dtype=float)

    fig, ax_lead = plt.subplots(figsize=(5.2, 3.1))

    bar_colors = ["#6baed6"] * len(rows)
    highlight_idx: int | None = None
    if highlight is not None:
        highlight_pct = highlight * 100.0
        for idx, value in enumerate(targets):
            if abs(value - highlight_pct) < 1e-6:
                highlight_idx = idx
                bar_colors[idx] = "#2171b5"
                break

    bars = ax_lead.bar(targets, lead_mean, width=0.6, color=bar_colors, label="Mean lead (steps)")
    ax_lead.set_xlabel("Guardrail target (%)")
    ax_lead.set_ylabel("Mean lead (steps)")
    ax_lead.set_xticks(targets)
    ax_lead.set_ylim(0, max(lead_mean.max() * 1.2, 4))
    ax_lead.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)

    ax_p = ax_lead.twinx()
    (line,) = ax_p.plot(
        targets,
        p_min,
        color="#d94801",
        marker="o",
        linewidth=2.0,
        label="Minimum permutation $p$-value",
    )
    ax_p.set_ylabel("Minimum permutation $p$-value")
    ax_p.set_ylim(0, max(1.02 * p_min.max(), 0.12))
    threshold = ax_p.axhline(0.05, color="#7f7f7f", linestyle="--", linewidth=1.0, label="$p=0.05$ threshold")

    if highlight_idx is not None:
        ax_p.scatter(
            [targets[highlight_idx]],
            [p_min[highlight_idx]],
            color="#d94801",
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
            s=60,
        )
        annotation = (
            f"{targets[highlight_idx]:.1f}% target\n"
            f"coverage {coverage[highlight_idx]:.2f}%\n"
            f"lead {lead_mean[highlight_idx]:.1f} steps\n"
            rf"$p_{{\min}}$ = {p_min[highlight_idx]:.3f}"
        )
        ax_p.annotate(
            annotation,
            xy=(targets[highlight_idx], p_min[highlight_idx]),
            xytext=(targets[highlight_idx] + 1.2, p_min[highlight_idx] + 0.18),
            arrowprops=dict(arrowstyle="->", color="#444444", linewidth=0.8),
            fontsize=9,
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbbbbb", linewidth=0.8),
        )

    handles = [bars, line, threshold]
    labels = [h.get_label() for h in handles]
    ax_lead.legend(handles, labels, loc="upper left", frameon=False)

    if title:
        ax_lead.set_title(title)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot guardrail sweep overlay")
    parser.add_argument("summary", type=Path, help="Sweep summary JSON produced by guardrail_sweep.py")
    parser.add_argument("output", type=Path, help="Output PNG path")
    parser.add_argument("--highlight", type=float, default=0.025, help="Guardrail target to highlight (fraction)")
    parser.add_argument("--title", type=str, default=None, help="Optional plot title")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_summary(args.summary)
    build_plot(rows, highlight=args.highlight, output_path=args.output, title=args.title)


if __name__ == "__main__":
    main()
