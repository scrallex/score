#!/usr/bin/env python3
"""Visualise semantic vs structural alignment for STM manifolds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sep_text_manifold.semantic import EmbeddingConfig, SemanticEmbedder, seed_similarity


def load_state(path: Path) -> Dict[str, object]:
    data = json.loads(path.read_text())
    if "string_scores" not in data:
        raise ValueError("State file missing 'string_scores'.")
    return data


def build_dataframe(
    state: Dict[str, object],
    *,
    seeds: Sequence[str],
    embedder: SemanticEmbedder,
    min_occurrences: int,
) -> pd.DataFrame:
    rows = []
    scores: Dict[str, Dict[str, object]] = state["string_scores"]  # type: ignore[assignment]
    strings = []
    payloads = []
    for string, payload in scores.items():
        occ = int(payload.get("occurrences", 0))
        if occ < min_occurrences:
            continue
        strings.append(string)
        payloads.append(payload)
    semantic = seed_similarity(strings, embedder=embedder, seeds=seeds)
    for string, payload, sem in zip(strings, payloads, semantic):
        metrics = payload.get("metrics", {})
        rows.append(
            {
                "string": string,
                "patternability": float(payload.get("patternability", 0.0)),
                "semantic_similarity": float(sem),
                "occurrences": int(payload.get("occurrences", 0)),
                "coherence": float(metrics.get("coherence", 0.0)),
                "entropy": float(metrics.get("entropy", 0.0)),
                "stability": float(metrics.get("stability", 0.0)),
                "rupture": float(metrics.get("rupture", 0.0)),
            }
        )
    if not rows:
        raise ValueError("No strings remain after filtering; lower --min-occurrences.")
    return pd.DataFrame(rows)


def plot_bridge(df: pd.DataFrame, *, seeds: Sequence[str], output: Path, top_labels: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    sizes = 40 + 10 * np.sqrt(df["occurrences"].to_numpy())
    scatter = ax.scatter(
        df["patternability"],
        df["semantic_similarity"],
        c=df["coherence"],
        s=sizes,
        cmap="viridis",
        alpha=0.7,
        edgecolors="none",
    )
    ax.set_xlabel("Structural patternability")
    ax.set_ylabel("Semantic similarity to seeds")
    ax.set_title("Semantic vs structural alignment\nSeeds: " + ", ".join(seeds))
    cbar = fig.colorbar(scatter, ax=ax, label="Mean coherence")
    cbar.ax.tick_params(labelsize=9)

    combined = df.assign(combined=lambda frame: 0.5 * frame["patternability"] + 0.5 * frame["semantic_similarity"])
    top = combined.sort_values("combined", ascending=False).head(top_labels)
    for _, row in top.iterrows():
        ax.annotate(
            row["string"],
            (row["patternability"], row["semantic_similarity"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            alpha=0.8,
        )

    ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path, help="STM state with string_scores")
    parser.add_argument("--seeds", nargs="+", required=True, help="Anchor phrases for semantic similarity")
    parser.add_argument("--min-occurrences", type=int, default=3, help="Minimum occurrences required per string")
    parser.add_argument("--top-labels", type=int, default=15, help="How many highest combined-score strings to annotate")
    parser.add_argument("--output", type=Path, default=Path("results/semantic_bridge_scatter.png"))
    parser.add_argument("--embedding-method", choices=["auto", "transformer", "hash"], default="auto")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--hash-dims", type=int, default=256, help="Dimensions when using hash fallback")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = load_state(args.state)
    embedder = SemanticEmbedder(EmbeddingConfig(method=args.embedding_method, model_name=args.model, dims=args.hash_dims))
    df = build_dataframe(state, seeds=args.seeds, embedder=embedder, min_occurrences=args.min_occurrences)
    plot_bridge(df, seeds=args.seeds, output=args.output, top_labels=args.top_labels)


if __name__ == "__main__":
    main()

