#!/usr/bin/env python3
"""Run threshold sweeps over a truth-pack + span set."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List

from reality_filter import SimSpanSource, TruthPackEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--spans", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/sweeps/output.csv"))
    parser.add_argument("--r-grid", nargs="*", type=int, default=[1, 2, 3])
    parser.add_argument("--lambda-grid", nargs="*", type=float, default=[0.12, 0.15, 0.18, 0.22, 0.25])
    parser.add_argument("--sigma-grid", nargs="*", type=float, default=[0.15, 0.2, 0.25, 0.3])
    parser.add_argument("--embedding-method", choices=["auto", "transformer", "hash"], default="hash")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--hash-dims", type=int, default=256)
    parser.add_argument("--structural-threshold", type=float, default=0.46)
    return parser.parse_args()


def load_spans(path: Path) -> List[str]:
    source = SimSpanSource(path)
    return [record.span for record in source]


def evaluate(
    engine: TruthPackEngine,
    spans: Iterable[str],
    *,
    r_min: int,
    hazard_max: float,
    sigma_min: float,
    structural_threshold: float,
) -> dict:
    total = 0
    admitted = 0
    repair_candidates = 0
    for span in spans:
        result = engine.evaluate_span(
            span,
            semantic_threshold=sigma_min,
            structural_threshold=structural_threshold,
            r_min=r_min,
            hazard_max=hazard_max,
            sigma_min=sigma_min,
        )
        total += 1
        if result.admitted:
            admitted += 1
        elif result.repair_candidate:
            repair_candidates += 1
    coverage = admitted / total if total else 0.0
    return {
        "total": total,
        "admitted": admitted,
        "coverage": coverage,
        "repair_candidates": repair_candidates,
    }


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
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["r_min", "lambda_max", "sigma_min", "total", "admitted", "coverage", "repair_candidates"])
        for r_min in args.r_grid:
            for lam in args.lambda_grid:
                for sigma in args.sigma_grid:
                    metrics = evaluate(
                        engine,
                        spans,
                        r_min=r_min,
                        hazard_max=lam,
                        sigma_min=sigma,
                        structural_threshold=args.structural_threshold,
                    )
                    writer.writerow([
                        r_min,
                        lam,
                        sigma,
                        metrics["total"],
                        metrics["admitted"],
                        metrics["coverage"],
                        metrics["repair_candidates"],
                    ])

    print(f"Wrote sweep results to {output_path}")


if __name__ == "__main__":
    main()
