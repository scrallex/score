#!/usr/bin/env python3
"""Generate synthetic PlanBench trace JSONs without requiring VAL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from val_to_trace import extract_init_atoms, read_plan


State = List[str]


def _parse_action(step: str) -> Tuple[str, List[str]]:
    text = step.strip().strip("()")
    parts = text.split()
    if not parts:
        raise ValueError(f"Unable to parse action from '{step}'")
    return parts[0].lower(), [p.lower() for p in parts[1:]]


def _apply_blocksworld(name: str, args: Sequence[str]) -> Tuple[List[str], List[str]]:
    if name == "pick-up" and len(args) == 1:
        x = args[0]
        return [f"holding {x}"], [f"on-table {x}", f"clear {x}", "handempty"]
    if name == "put-down" and len(args) == 1:
        x = args[0]
        return [f"on-table {x}", f"clear {x}", "handempty"], [f"holding {x}"]
    if name == "stack" and len(args) == 2:
        x, y = args
        return [f"on {x} {y}", f"clear {x}", "handempty"], [f"holding {x}", f"clear {y}"]
    if name == "unstack" and len(args) == 2:
        x, y = args
        return [f"holding {x}", f"clear {y}"], [f"on {x} {y}", f"clear {x}", "handempty"]
    return [], []


def _apply_mystery(name: str, args: Sequence[str]) -> Tuple[List[str], List[str]]:
    if name == "lift" and len(args) == 1:
        x = args[0]
        return [f"holding {x}"], [f"grounded {x}", f"free {x}", "idle"]
    if name == "drop" and len(args) == 1:
        x = args[0]
        return [f"grounded {x}", f"free {x}", "idle"], [f"holding {x}"]
    if name == "bind" and len(args) == 2:
        x, y = args
        return [f"rel {x} {y}", f"free {x}", "idle"], [f"holding {x}", f"free {y}"]
    return [], []


def _apply_logistics(name: str, args: Sequence[str]) -> Tuple[List[str], List[str]]:
    if name == "load-truck" and len(args) == 3:
        pkg, trk, loc = args
        return [f"in {pkg} {trk}"], [f"at {pkg} {loc}"]
    if name == "unload-truck" and len(args) == 3:
        pkg, trk, loc = args
        return [f"at {pkg} {loc}"], [f"in {pkg} {trk}"]
    if name == "drive" and len(args) == 3:
        trk, src, dst = args
        return [f"at-vehicle {trk} {dst}"], [f"at-vehicle {trk} {src}"]
    if name == "load-plane" and len(args) == 3:
        pkg, pln, loc = args
        return [f"in {pkg} {pln}"], [f"at {pkg} {loc}"]
    if name == "unload-plane" and len(args) == 3:
        pkg, pln, loc = args
        return [f"at {pkg} {loc}"], [f"in {pkg} {pln}"]
    if name == "fly" and len(args) == 3:
        pln, src, dst = args
        return [f"at-vehicle {pln} {dst}"], [f"at-vehicle {pln} {src}"]
    return [], []


DOMAIN_EFFECTS = {
    "blocksworld": _apply_blocksworld,
    "mystery_bw": _apply_mystery,
    "logistics": _apply_logistics,
}


def simulate_trace(
    domain: str,
    init_atoms: Iterable[str],
    actions: Sequence[str],
    *,
    failure_step: Optional[int] = None,
) -> Tuple[List[Dict[str, object]], bool, List[str], Optional[int]]:
    apply_effects = DOMAIN_EFFECTS[domain]
    state = {atom.lower() for atom in init_atoms}
    transitions: List[Dict[str, object]] = []
    errors: List[str] = []
    status_ok = True
    failed_at: Optional[int] = None

    for idx, raw_action in enumerate(actions):
        name, args = _parse_action(raw_action)
        adds, dels = apply_effects(name, args)
        before = sorted(state)
        step_status = "valid"
        step_errors: List[str] = []
        if failure_step is not None and idx >= failure_step:
            step_status = "invalid"
            step_errors.append("synthetic failure: precondition violated")
            status_ok = False
            failed_at = idx
            next_state = sorted(state)
        else:
            for atom in dels:
                state.discard(atom.lower())
            for atom in adds:
                state.add(atom.lower())
            next_state = sorted(state)

        transitions.append(
            {
                "step": idx,
                "time": float(idx),
                "action": raw_action,
                "state": before,
                "effects": {
                    "del": sorted({atom.lower() for atom in dels}),
                    "add": sorted({atom.lower() for atom in adds}),
                },
                "status": step_status,
                "next_state": next_state,
                "errors": step_errors,
            }
        )

        if step_status == "invalid":
            errors.extend(step_errors)
            break

    return transitions, status_ok, errors, failed_at


def build_trace_payload(
    *,
    domain_dir: Path,
    domain: str,
    plan_path: Path,
    problem_path: Path,
    plan_type: str,
    failure_step: Optional[int],
) -> Dict[str, object]:
    init_atoms = extract_init_atoms(problem_path)
    actions = read_plan(plan_path)

    transitions, status_ok, errors, failed_at = simulate_trace(
        domain,
        init_atoms,
        actions,
        failure_step=failure_step,
    )

    payload: Dict[str, object] = {
        "domain": str(domain_dir / "domain.pddl"),
        "problem": str(problem_path),
        "plan": str(plan_path),
        "plan_type": plan_type,
        "initial_state": sorted(atom.lower() for atom in init_atoms),
        "status": "valid" if status_ok else "invalid",
        "exit_code": 0 if status_ok else 1,
        "validator_output": "synthetic",
        "transitions": transitions,
        "errors": errors,
        "failed_at_step": failed_at,
    }
    return payload


def generate_traces_for_domain(domain_dir: Path, *, overwrite: bool) -> None:
    domain = domain_dir.name
    if domain not in DOMAIN_EFFECTS:
        return

    problems_dir = domain_dir / "problems"
    plans_dir = domain_dir / "plans_valid"
    trace_dir = domain_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    if overwrite:
        for existing in trace_dir.glob("*.json"):
            existing.unlink()

    for plan_path in sorted(plans_dir.glob("*.txt")):
        problem_path = problems_dir / f"{plan_path.stem}.pddl"
        if not problem_path.exists():
            continue

        valid_payload = build_trace_payload(
            domain_dir=domain_dir,
            domain=domain,
            plan_path=plan_path,
            problem_path=problem_path,
            plan_type="valid",
            failure_step=None,
        )
        (trace_dir / f"{plan_path.stem}_valid.json").write_text(
            json.dumps(valid_payload, indent=2),
            encoding="utf-8",
        )

        actions = read_plan(plan_path)
        if not actions:
            continue
        failure_step = max(1, len(actions) // 2)
        corrupt_payload = build_trace_payload(
            domain_dir=domain_dir,
            domain=domain,
            plan_path=plan_path,
            problem_path=problem_path,
            plan_type="corrupt",
            failure_step=failure_step,
        )
        (trace_dir / f"{plan_path.stem}_corrupt.json").write_text(
            json.dumps(corrupt_payload, indent=2),
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetically derive PlanBench trace JSON files")
    parser.add_argument("--root", type=Path, default=Path("data/planbench_public"))
    parser.add_argument("--overwrite", action="store_true", help="Remove existing traces before generating new ones")
    args = parser.parse_args()

    root = args.root
    if not root.exists():
        raise FileNotFoundError(f"PlanBench root not found: {root}")

    for domain_dir in sorted(root.iterdir()):
        if not domain_dir.is_dir():
            continue
        generate_traces_for_domain(domain_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
