#!/usr/bin/env python3
"""Convert PDDL plans into STM-ready trace JSONs using VAL."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_VALIDATE = Path("external/VAL/build/bin/Validate")
RE_CHECK = re.compile(r"^Checking next happening \(time ([0-9.+-eE]+)\)")
RE_DEL = re.compile(r"^Deleting \((.*)\)")
RE_ADD = re.compile(r"^Adding \((.*)\)")
RE_FAIL = re.compile(r"^Plan failed because of (.*)")
RE_PLAN = re.compile(r"^\(.*\)$")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def strip_comments(text: str) -> str:
    return "\n".join(line.split(";", 1)[0] for line in text.splitlines())


def extract_init_atoms(problem_path: Path) -> List[str]:
    content = strip_comments(problem_path.read_text(encoding="utf-8"))
    lowered = content.lower()
    start = lowered.find("(:init")
    if start == -1:
        raise ValueError(f"(:init section not found in {problem_path}")
    depth = 0
    chunk: List[str] = []
    for ch in content[start:]:
        chunk.append(ch)
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                break
    init_text = "".join(chunk)
    atoms = [atom.strip()[1:-1] for atom in re.findall(r"\([^()]+\)", init_text)]
    return [" ".join(atom.split()) for atom in atoms if atom]


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


# ---------------------------------------------------------------------------
# VAL execution
# ---------------------------------------------------------------------------

def run_validate(validator: Path, domain: Path, problem: Path, plan: Path) -> Tuple[int, str]:
    cmd = [str(validator), "-v", str(domain), str(problem), str(plan)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def build_trace(
    init_atoms: Sequence[str],
    actions: Sequence[str],
    validate_output: str,
    exit_code: int,
) -> Tuple[List[Dict[str, Any]], bool, List[str], Optional[int]]:
    state = set(init_atoms)
    transitions: List[Dict[str, Any]] = []
    errors: List[str] = []
    status = exit_code == 0
    failed_step: Optional[int] = None

    lines = validate_output.splitlines()
    action_index = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        check_match = RE_CHECK.match(line)
        if check_match:
            if action_index >= len(actions):
                break
            time_value = float(check_match.group(1))
            action = actions[action_index]
            before_state = sorted(state)
            dels: List[str] = []
            adds: List[str] = []
            step_errors: List[str] = []
            step_status = "valid"
            i += 1
            while i < len(lines):
                cur = lines[i].strip()
                if not cur:
                    i += 1
                    continue
                if RE_CHECK.match(cur):
                    i -= 1
                    break
                del_match = RE_DEL.match(cur)
                add_match = RE_ADD.match(cur)
                fail_match = RE_FAIL.match(cur)
                if fail_match:
                    step_status = "invalid"
                    msg = fail_match.group(1).strip()
                    step_errors.append(msg)
                    j = i + 1
                    while j < len(lines):
                        candidate = lines[j].strip()
                        if candidate:
                            step_errors.append(candidate)
                            break
                        j += 1
                    errors.extend(step_errors)
                    while i < len(lines) and not lines[i].strip().startswith("Plan failed"):
                        i += 1
                    i -= 1
                    break
                if cur.startswith("Plan executed successfully") or cur.startswith("Plan valid") or cur.startswith("Plan Repair Advice") or cur.startswith("Plan failed") or cur.startswith("Goal not satisfied"):
                    i -= 1
                    break
                if del_match:
                    dels.append(del_match.group(1).strip())
                elif add_match:
                    adds.append(add_match.group(1).strip())
                i += 1
            next_state = set(state)
            for atom in dels:
                next_state.discard(atom)
            for atom in adds:
                next_state.add(atom)
            transitions.append(
                {
                    "step": action_index,
                    "time": time_value,
                    "action": action,
                    "state": before_state,
                    "effects": {"del": dels, "add": adds},
                    "status": step_status,
                    "next_state": sorted(next_state if step_status == "valid" else state),
                    "errors": step_errors,
                }
            )
            if step_status == "valid":
                state = next_state
            else:
                status = False
                failed_step = action_index
                break
            action_index += 1
        i += 1

    return transitions, status, errors, failed_step


def generate_trace(
    domain: Path,
    problem: Path,
    plan: Path,
    output: Path,
    validator: Path,
    plan_type: Optional[str] = None,
) -> Dict[str, Any]:
    init_atoms = extract_init_atoms(problem)
    actions = read_plan(plan)
    exit_code, val_output = run_validate(validator, domain, problem, plan)
    transitions, status, errors, failed_step = build_trace(init_atoms, actions, val_output, exit_code)
    trace = {
        "domain": str(domain),
        "problem": str(problem),
        "plan": str(plan),
        "initial_state": sorted(init_atoms),
        "status": "valid" if status else "invalid",
        "exit_code": exit_code,
        "validator_output": val_output,
        "transitions": transitions,
        "errors": errors,
        "failed_at_step": failed_step,
    }
    if plan_type:
        trace["plan_type"] = plan_type
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    return trace


# ---------------------------------------------------------------------------
# Root-mode utilities
# ---------------------------------------------------------------------------

def _infer_problem(plan: Path, problems_dir: Path) -> Path:
    candidates = list(problems_dir.glob("*.pddl"))
    if not candidates:
        raise FileNotFoundError(f"No problem files found in {problems_dir}")
    # Try matching by shared digits
    digits = "".join(ch for ch in plan.stem if ch.isdigit())
    if digits:
        filtered = [p for p in candidates if digits in p.stem]
        if len(filtered) == 1:
            return filtered[0]
        if len(filtered) > 1:
            candidates = filtered
    # Try matching by stem tokens
    tokens = [token for token in re.split(r"[^a-zA-Z0-9]", plan.stem) if token]
    if tokens:
        filtered = [p for p in candidates if any(token in p.stem for token in tokens)]
        if len(filtered) == 1:
            return filtered[0]
        if filtered:
            candidates = filtered
    # Fallback: single problem
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"Unable to infer problem for {plan} in {problems_dir}")


def process_root(
    root: Path,
    domains: Sequence[str],
    validator: Path,
    problems_dirname: str,
    valid_dirname: str,
    corrupt_dirname: str,
    traces_dirname: str,
    plan_suffix: str,
) -> None:
    for domain_name in domains:
        domain_root = root / domain_name
        domain_file = domain_root / "domain.pddl"
        if not domain_file.exists():
            raise FileNotFoundError(f"domain.pddl not found for {domain_name}")
        problems_dir = domain_root / problems_dirname
        valid_dir = domain_root / valid_dirname
        corrupt_dir = domain_root / corrupt_dirname
        traces_dir = domain_root / traces_dirname
        traces_dir.mkdir(parents=True, exist_ok=True)

        for plan_dir, plan_type in ((valid_dir, "valid"), (corrupt_dir, "corrupt")):
            if not plan_dir.exists():
                continue
            for plan_path in sorted(plan_dir.glob(plan_suffix)):
                problem_path = _infer_problem(plan_path, problems_dir)
                output_path = traces_dir / f"{plan_path.stem}.trace.json"
                trace = generate_trace(domain_file, problem_path, plan_path, output_path, validator, plan_type)
                failed_step = trace.get("failed_at_step")
                print(
                    f"[{domain_name}] {plan_path.name} → {output_path.name} "
                    f"status={trace['status']} failed_at_step={failed_step}"
                )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate VAL trace JSONs")
    parser.add_argument("domain", nargs="?", type=Path)
    parser.add_argument("problem", nargs="?", type=Path)
    parser.add_argument("plan", nargs="?", type=Path)
    parser.add_argument("--output", type=Path, help="Output JSON path")
    parser.add_argument("--validator", type=Path, default=DEFAULT_VALIDATE)
    parser.add_argument("--plan-type", choices=("valid", "corrupt"), help="Tag written into the trace JSON")
    parser.add_argument("--root", type=Path, help="Root directory containing PlanBench-style domains")
    parser.add_argument("--domains", type=str, help="Comma-separated list of domains under --root")
    parser.add_argument("--problems-dir", default="problems")
    parser.add_argument("--plans-valid", default="plans_valid")
    parser.add_argument("--plans-corrupt", default="plans_corrupt")
    parser.add_argument("--traces-dir", default="traces")
    parser.add_argument("--plan-glob", default="*.txt")
    args = parser.parse_args()

    if args.root:
        domain_list = (
            [d.strip() for d in args.domains.split(",") if d.strip()]
            if args.domains
            else [d.name for d in sorted(args.root.iterdir()) if d.is_dir()]
        )
        process_root(
            root=args.root,
            domains=domain_list,
            validator=args.validator,
            problems_dirname=args.problems_dir,
            valid_dirname=args.plans_valid,
            corrupt_dirname=args.plans_corrupt,
            traces_dirname=args.traces_dir,
            plan_suffix=args.plan_glob,
        )
        return

    if not (args.domain and args.problem and args.plan):
        parser.error("domain, problem, and plan must be provided unless --root is used")

    output = args.output or args.plan.with_suffix(args.plan.suffix + ".trace.json")
    trace = generate_trace(args.domain, args.problem, args.plan, output, args.validator, args.plan_type)
    print(f"Wrote {output} (status={trace['status']} failed_at_step={trace['failed_at_step']})")


if __name__ == "__main__":
    main()
