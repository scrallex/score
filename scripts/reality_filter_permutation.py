#!/usr/bin/env python3
"""Permutation-style significance estimate for reality filter coverage."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from reality_filter import SimSpanSource, TruthPackEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--spans", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--r-min", type=int, default=2)
    parser.add_argument("--lambda-max", type=float, default=0.25)
    parser.add_argument("--sigma-min", type=float, default=0.25)
    parser.add_argument("--structural-threshold", type=float, default=0.46)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("results/permutation.json"))
    parser.add_argument("--embedding-method", choices=["auto", "transformer", "hash"], default="hash")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--hash-dims", type=int, default=256)
    return parser.parse_args()


def load_spans(path: Path) -> list[str]:
    source = SimSpanSource(path)
    return [record.span for record in source]


def main() -> None:
    args = parse_args()
    engine = TruthPackEngine.from_manifest(
        args.manifest,
        seeds=[],
        embedding_method=args.embedding_method,
        model_name=args.model,
        hash_dims=args.hash_dims,
    )
    spans = load_spans(args.spans)

    evaluations = [
        engine.evaluate_span(
            span,
            semantic_threshold=args.sigma_min,
            structural_threshold=args.structural_threshold,
            r_min=args.r_min,
            hazard_max=args.lambda_max,
            sigma_min=args.sigma_min,
        )
        for span in spans
    ]

    total = len(evaluations)
    admitted = sum(1 for e in evaluations if e.admitted)
    observed_coverage = admitted / total if total else 0.0

    rng = random.Random(args.seed)
    successes = 0
    for _ in range(args.iterations):
        simulated = sum(1 for _ in range(total) if rng.random() < observed_coverage)
        if simulated >= admitted:
            successes += 1

    p_value = successes / args.iterations if args.iterations else 1.0
    payload = {
        "manifest": str(args.manifest),
        "spans": str(args.spans),
        "iterations": args.iterations,
        "observed_admitted": admitted,
        "observed_total": total,
        "observed_coverage": observed_coverage,
        "p_value": p_value,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Permutation summary written to {args.output}")


if __name__ == "__main__":
    main()
