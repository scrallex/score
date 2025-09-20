#!/usr/bin/env python3
"""Inject a delayed error into a plan by perturbing actions near the tail."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import List


def read_plan(plan_path: Path) -> List[str]:
    actions: List[str] = []
    for raw in plan_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        actions.append(" ".join(line.split()))
    if not actions:
        raise ValueError(f"No actions found in {plan_path}")
    return actions


def corrupt_plan(actions: List[str], *, seed: int | None) -> List[str]:
    rng = random.Random(seed)
    n = len(actions)
    if n < 2:
        raise ValueError("Plan too short to corrupt")
    low = max(0, math.floor(0.4 * n))
    high = max(low, min(n - 1, math.floor(0.8 * n)))
    idx = rng.randint(low, high)
    mutated = actions.copy()
    mutated.pop(idx)
    return mutated


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject a delayed error by removing an action")
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    actions = read_plan(args.plan)
    corrupted = corrupt_plan(actions, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(corrupted) + "\n")
    print(f"Wrote corrupted plan → {args.output}")


if __name__ == "__main__":
    main()
