#!/usr/bin/env python3
"""Bridge structural manifold metrics with semantic embeddings.

This helper script demonstrates how semantic proximity and structural
patternability can be combined.  It loads an STM state file, projects
the strings into a semantic embedding space, and contrasts the
semantics-driven rankings with the existing structural metrics.  The
output highlights where the manifold already sees repetition and where
semantic cues surface additional context.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from sep_text_manifold.semantic import EmbeddingConfig, SemanticEmbedder, seed_similarity


def _load_state(path: Path) -> Dict[str, object]:
    data = json.loads(path.read_text())
    if "string_scores" not in data:
        raise ValueError("State file does not contain 'string_scores'. Re-run ingest with --store-signals.")
    return data


def _select_strings(
    state: Dict[str, object],
    *,
    limit: int,
    min_occurrences: int,
) -> List[Tuple[str, Dict[str, object]]]:
    scores: Dict[str, Dict[str, object]] = state["string_scores"]  # type: ignore[assignment]
    filtered = [
        (string, payload)
        for string, payload in scores.items()
        if payload.get("occurrences", 0) >= min_occurrences
    ]
    filtered.sort(key=lambda item: item[1].get("occurrences", 0), reverse=True)
    if limit > 0:
        filtered = filtered[:limit]
    return filtered


def _summary_row(string: str, payload: Dict[str, object], semantic_score: float) -> Dict[str, object]:
    metrics = payload.get("metrics", {})
    return {
        "string": string,
        "occurrences": payload.get("occurrences", 0),
        "patternability": payload.get("patternability", 0.0),
        "semantic_similarity": semantic_score,
        "coherence": metrics.get("coherence", 0.0),
        "entropy": metrics.get("entropy", 0.0),
        "stability": metrics.get("stability", 0.0),
        "rupture": metrics.get("rupture", 0.0),
    }


def run_demo(
    *,
    state_path: Path,
    seeds: Sequence[str],
    top_k: int,
    min_occurrences: int,
    embedding_method: str,
    model_name: str,
    hash_dims: int,
) -> Dict[str, object]:
    state = _load_state(state_path)
    selected = _select_strings(state, limit=5000, min_occurrences=min_occurrences)
    if not selected:
        raise ValueError("No strings pass the occurrence threshold; relax filters or re-run ingest.")
    strings = [item[0] for item in selected]
    payloads = [item[1] for item in selected]

    embedder = SemanticEmbedder(EmbeddingConfig(method=embedding_method, model_name=model_name, dims=hash_dims))
    semantic_scores = seed_similarity(strings, embedder=embedder, seeds=seeds)

    rows = [_summary_row(string, payload, float(score)) for string, payload, score in zip(strings, payloads, semantic_scores)]

    by_structure = sorted(rows, key=lambda row: row["patternability"], reverse=True)[:top_k]
    by_semantics = sorted(rows, key=lambda row: row["semantic_similarity"], reverse=True)[:top_k]

    combined = []
    for row in rows:
        combined_score = 0.5 * row["patternability"] + 0.5 * max(0.0, row["semantic_similarity"])
        entry = dict(row)
        entry["combined_score"] = combined_score
        combined.append(entry)
    by_combined = sorted(combined, key=lambda row: row["combined_score"], reverse=True)[:top_k]

    structural = np.array([row["patternability"] for row in rows], dtype=np.float32)
    semantic = np.array([row["semantic_similarity"] for row in rows], dtype=np.float32)
    structural_mean = float(structural.mean())
    semantic_mean = float(semantic.mean())
    covariance = float(np.cov(structural, semantic)[0, 1])
    if structural.std() > 0 and semantic.std() > 0:
        corr = float(np.corrcoef(structural, semantic)[0, 1])
    else:
        corr = 0.0

    return {
        "seeds": list(seeds),
        "counts": {
            "total_strings": len(rows),
            "min_occurrences": min_occurrences,
        },
        "statistics": {
            "patternability_mean": structural_mean,
            "semantic_mean": semantic_mean,
            "covariance": covariance,
            "pearson_r": corr,
        },
        "top_structural": by_structure,
        "top_semantic": by_semantics,
        "top_combined": by_combined,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path, help="STM state JSON with string_scores")
    parser.add_argument("--seeds", nargs="+", required=True, help="Seed strings or phrases for semantic comparison")
    parser.add_argument("--top-k", type=int, default=10, help="Number of rows to show per ranking")
    parser.add_argument("--min-occurrences", type=int, default=2, help="Minimum string occurrences to keep")
    parser.add_argument(
        "--embedding-method",
        choices=["auto", "transformer", "hash"],
        default="auto",
        help="Embedding backend: transformer when available, deterministic hash otherwise",
    )
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--hash-dims", type=int, default=256, help="Dimensions for the hash fallback")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    report = run_demo(
        state_path=args.state,
        seeds=args.seeds,
        top_k=args.top_k,
        min_occurrences=args.min_occurrences,
        embedding_method=args.embedding_method,
        model_name=args.model,
        hash_dims=args.hash_dims,
    )
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload)


if __name__ == "__main__":
    main()

