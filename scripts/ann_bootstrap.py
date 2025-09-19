#!/usr/bin/env python3
"""Bootstrap confidence interval for ANN distances in twin matches."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap CI for ANN distances")
    parser.add_argument("twins", type=Path, help="Twin JSON file")
    parser.add_argument("--min-windows", type=int, default=50, help="Minimum aligned windows threshold")
    parser.add_argument("--iterations", type=int, default=2000, help="Bootstrap iterations")
    parser.add_argument("--seed", type=int, help="Random seed")
    return parser.parse_args()


def collect_distances(payload: dict, min_windows: int) -> List[float]:
    distances: List[float] = []
    for result in payload.get("results", []):
        matches = result.get("matches", [])
        if len(matches) < min_windows:
            continue
        for match in matches:
            if "distance" in match:
                distances.append(float(match["distance"]))
    return distances


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    payload = json.loads(args.twins.read_text(encoding="utf-8"))
    distances = collect_distances(payload, args.min_windows)
    if not distances:
        raise SystemExit("No distances found meeting the min-windows criterion")

    B = args.iterations
    samples = np.asarray(distances)
    boot_means = np.empty(B, dtype=float)
    for i in range(B):
        resample = np.random.choice(samples, size=samples.size, replace=True)
        boot_means[i] = resample.mean()

    ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])
    result = {
        "mean_ann_distance": float(samples.mean()),
        "ci95_lower": float(ci_lower),
        "ci95_upper": float(ci_upper),
        "iterations": B,
        "sample_size": int(samples.size),
    }
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    import sys

    main()
