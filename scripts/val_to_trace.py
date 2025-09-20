#!/usr/bin/env python3
"""Convert a (domain, problem, plan) triple into an STM-ready trace JSON via VAL."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


DEFAULT_VALIDATE = Path("external/VAL/build/bin/Validate")
RE_CHECK = re.compile(r"^Checking next happening \(time ([0-9.+-eE]+)\)")
RE_DEL = re.compile(r"^Deleting \((.*)\)")
RE_ADD = re.compile(r"^Adding \((.*)\)")
RE_FAIL = re.compile(r"^Plan failed because of (.*)")
RE_PLAN = re.compile(r"^\(.*\)$")


def strip_comments(text: str) -> str:
    lines: List[str] = []
    for line in text.splitlines():
        if ";" in line:
            line = line.split(";", 1)[0]
        lines.append(line)
    return "\n".join(lines)


def extract_init_atoms(problem_path: Path) -> List[str]:
    content = strip_comments(problem_path.read_text(encoding="utf-8"))
    lowered = content.lower()
    start = lowered.find("(:init")
    if start == -1:
        raise ValueError(f"(:init section not found in {problem_path}")
    idx = start
    depth = 0
    init_chunk: List[str] = []
    while idx < len(content):
        ch = content[idx]
        init_chunk.append(ch)
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                break
        idx += 1
    init_text = "".join(init_chunk)
    atoms = re.findall(r"\([^()]+\)", init_text)
    cleaned: List[str] = []
    for atom in atoms:
        atom = " ".join(atom.strip()[1:-1].split())
        if atom:
            cleaned.append(atom)
    return cleaned


def read_plan(plan_path: Path) -> List[str]:
    actions: List[str] = []
    for raw in plan_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        actions.append(" ".join(line.split()))
    if not actions:
        raise ValueError(f"No actions found in plan {plan_path}")
    return actions


def run_validate(validate_bin: Path, domain: Path, problem: Path, plan: Path) -> Tuple[int, str]:
    cmd = [str(validate_bin), "-v", str(domain), str(problem), str(plan)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def build_trace(
    init_atoms: Sequence[str],
    actions: Sequence[str],
    validate_output: str,
    exit_code: int,
) -> Tuple[List[dict], bool, List[str]]:
    state = set(init_atoms)
    transitions: List[dict] = []
    errors: List[str] = []
    status = exit_code == 0

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
                break
            action_index += 1
        i += 1

    return transitions, status, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate VAL trace JSON for STM inputs")
    parser.add_argument("domain", type=Path)
    parser.add_argument("problem", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=Path, help="Output JSON path (default: plan path + .trace.json)")
    parser.add_argument(
        "--validator",
        type=Path,
        default=DEFAULT_VALIDATE,
        help="Path to VAL 'Validate' binary",
    )
    args = parser.parse_args()

    if not args.validator.exists():
        raise FileNotFoundError(f"Validate binary not found at {args.validator}")

    init_atoms = extract_init_atoms(args.problem)
    actions = read_plan(args.plan)
    exit_code, out_text = run_validate(args.validator, args.domain, args.problem, args.plan)
    transitions, status, errors = build_trace(init_atoms, actions, out_text, exit_code)

    trace = {
        "domain": str(args.domain),
        "problem": str(args.problem),
        "plan": str(args.plan),
        "initial_state": sorted(init_atoms),
        "status": "valid" if status else "invalid",
        "exit_code": exit_code,
        "validator_output": out_text,
        "transitions": transitions,
        "errors": errors,
    }

    output_path = args.output or (args.plan.with_suffix(args.plan.suffix + ".trace.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
