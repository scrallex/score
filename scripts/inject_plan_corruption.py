#!/usr/bin/env python3
"""Inject delayed precondition failures into plans by removing mid-tail actions."""

from __future__ import annotations

import argparse
import math
import random
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from val_to_trace import (
    build_trace,
    extract_init_atoms,
    read_plan,
    run_validate,
)


def mutate_plan(
    actions: Sequence[str],
    rng: random.Random,
    min_frac: float,
    max_frac: float,
) -> Tuple[List[str], int]:
    n = len(actions)
    invalid = actions[-1]
    m = n + 1
    low = max(1, math.floor(min_frac * m))
    high = min(m - 2, math.floor(max_frac * m))
    if high < low:
        high = low
    if high <= low:
        idx = min(low, m - 2)
    else:
        idx = rng.randint(low, high)
    mutated = list(actions)
    mutated.insert(idx, invalid)
    return mutated, idx


def evaluate_mutation(
    validator: Path,
    domain: Path,
    problem: Path,
    mutated_actions: Sequence[str],
    init_atoms: Sequence[str],
    temp_plan_path: Path,
) -> Tuple[bool, Optional[int]]:
    temp_plan_path.write_text("\n".join(mutated_actions) + "\n", encoding="utf-8")
    exit_code, val_output = run_validate(validator, domain, problem, temp_plan_path)
    transitions, status, _errors, failed_step = build_trace(init_atoms, mutated_actions, val_output, exit_code)
    if not status and failed_step is None and transitions:
        failed_step = transitions[-1]["step"]
    return status, failed_step


def generate_corruption(
    validator: Path,
    domain: Path,
    problem: Path,
    source_plan: Path,
    output_plan: Path,
    min_frac: float,
    max_frac: float,
    max_retries: int,
    base_seed: Optional[int],
) -> Tuple[bool, Optional[int], int]:
    rng = random.Random(base_seed)
    actions = read_plan(source_plan)
    init_atoms = extract_init_atoms(problem)
    min_index = max(1, math.floor(min_frac * (len(actions) + 1)))

    best_failed_step: Optional[int] = None
    success = False
    attempts = 0

    for attempt in range(max_retries):
        attempts = attempt + 1
        mutated_actions, _ = mutate_plan(actions, rng, min_frac, max_frac)
        with tempfile.NamedTemporaryFile("w", delete=False, dir=output_plan.parent, suffix=output_plan.suffix) as tmp:
            temp_path = Path(tmp.name)
        status_ok, failed_step = evaluate_mutation(
            validator, domain, problem, mutated_actions, init_atoms, temp_path
        )
        if not status_ok:
            best_failed_step = failed_step
            if failed_step is not None and failed_step >= min_index:
                shutil.move(str(temp_path), output_plan)
                success = True
                break
        temp_path.unlink(missing_ok=True)
    if not success:
        raise RuntimeError(
            f"Failed to inject delayed error for {source_plan} after {max_retries} attempts"
        )
    return success, best_failed_step, attempts


def process_single(args: argparse.Namespace) -> None:
    output_plan = args.output
    output_plan.parent.mkdir(parents=True, exist_ok=True)
    success, failed_step, attempts = generate_corruption(
        validator=args.validator,
        domain=args.domain,
        problem=args.problem,
        source_plan=args.plan,
        output_plan=output_plan,
        min_frac=args.min_frac,
        max_frac=args.max_frac,
        max_retries=args.max_retries,
        base_seed=args.seed,
    )
    status = "ok" if success else "fallback"
    print(
        f"Corrupted {args.plan.name} → {output_plan} ({status}) "
        f"failed_at_step={failed_step} attempts={attempts}"
    )


def _infer_problem(plan: Path, problems_dir: Path) -> Path:
    from val_to_trace import _infer_problem as infer  # reuse heuristic

    return infer(plan, problems_dir)


def process_root(args: argparse.Namespace) -> None:
    root = args.root
    domain_names = (
        [d.strip() for d in args.domains.split(",") if d.strip()]
        if args.domains
        else [d.name for d in sorted(root.iterdir()) if d.is_dir()]
    )
    rng = random.Random(args.seed)

    for domain_name in domain_names:
        domain_dir = root / domain_name
        domain_file = domain_dir / "domain.pddl"
        problems_dir = domain_dir / args.problems_dir
        valid_dir = domain_dir / args.plans_valid
        corrupt_dir = domain_dir / args.plans_corrupt
        corrupt_dir.mkdir(parents=True, exist_ok=True)

        for plan_path in sorted(valid_dir.glob(args.plan_glob)):
            problem_path = _infer_problem(plan_path, problems_dir)
            output_plan = corrupt_dir / f"{plan_path.stem}_corrupt{plan_path.suffix}"
            seed = rng.randint(0, 1_000_000)
            success, failed_step, attempts = generate_corruption(
                validator=args.validator,
                domain=domain_file,
                problem=problem_path,
                source_plan=plan_path,
                output_plan=output_plan,
                min_frac=args.min_frac,
                max_frac=args.max_frac,
                max_retries=args.max_retries,
                base_seed=seed,
            )
            status = "ok" if success else "fallback"
            print(
                f"[{domain_name}] {plan_path.name} → {output_plan.name} ({status}) "
                f"failed_at_step={failed_step} attempts={attempts}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inject delayed failures into plans")
    parser.add_argument("plan", nargs="?", type=Path)
    parser.add_argument("--problem", type=Path, help="Problem file (single-plan mode)")
    parser.add_argument("--domain", type=Path, help="Domain file (single-plan mode)")
    parser.add_argument("--output", type=Path, help="Output corrupted plan path")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--validator", type=Path, default=Path("external/VAL/build/bin/Validate"))
    parser.add_argument("--min-frac", type=float, default=0.4)
    parser.add_argument("--max-frac", type=float, default=0.85)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--root", type=Path, help="Root directory containing PlanBench-style domains")
    parser.add_argument("--domains", type=str, help="Comma-separated domain list")
    parser.add_argument("--problems-dir", default="problems")
    parser.add_argument("--plans-valid", default="plans_valid")
    parser.add_argument("--plans-corrupt", default="plans_corrupt")
    parser.add_argument("--plan-glob", default="*.txt")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.root:
        process_root(args)
        return

    if not (args.plan and args.problem and args.domain and args.output):
        parser.error("plan, problem, domain, and --output are required in single-plan mode")

    process_single(args)


if __name__ == "__main__":
    main()
