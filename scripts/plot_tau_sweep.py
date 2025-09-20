#!/usr/bin/env python3
"""Generate τ-sweep plots for PlanBench and CodeTrace cohorts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "docs" / "note" / "appendix_tau_sweep.csv"
OUTPUT_DIR = REPO_ROOT / "docs" / "note"


def load_data() -> Dict[str, pd.DataFrame]:
    table = pd.read_csv(DATA_PATH)
    cohorts: Dict[str, List[pd.Series]] = {}
    for cohort, group in table.groupby("cohort"):
        cohorts[cohort] = [row for _, row in group.iterrows()]
    frames: Dict[str, pd.DataFrame] = {}
    for cohort, rows in cohorts.items():
        frames[cohort] = pd.DataFrame(rows)
    return frames


def plot_cohort(name: str, frame: pd.DataFrame) -> Path:
    output_path = OUTPUT_DIR / f"fig_tau_sweep_{name.lower()}.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(frame["tau"], frame["twin_rate"], marker="o", linewidth=2, color="#0D47A1")
    ax.set_title(f"τ-sweep twin rates — {name}")
    ax.set_xlabel("τ")
    ax.set_ylabel("Twin acceptance rate")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle=":", linewidth=0.5)
    for tau, rate in zip(frame["tau"], frame["twin_rate"], strict=False):
        ax.annotate(f"{rate:.2f}", (tau, rate), textcoords="offset points", xytext=(0, 8), ha="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    frames = load_data()
    for cohort, frame in frames.items():
        path = plot_cohort(cohort, frame.sort_values("tau"))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
