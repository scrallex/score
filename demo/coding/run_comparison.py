#!/usr/bin/env python3
"""Compare baseline vs STM traces across demo coding tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

from stm_adapters.code_trace_adapter import CodeTraceAdapter

TASK_ROOT = Path(__file__).resolve().parent / "tasks"
OUTPUT_ROOT = Path(__file__).resolve().parent / "output"
VARIANTS = ("baseline", "stm")


def load_steps(trace_path: Path) -> List[Mapping[str, object]]:
    text = trace_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if "\n" in text and not text.lstrip().startswith("{"):
        steps: List[Mapping[str, object]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if isinstance(data, Mapping):
                steps.append(data)
        return steps
    data = json.loads(text)
    if isinstance(data, Mapping) and "steps" in data:
        payload = data.get("steps")
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, Mapping)]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, Mapping)]
    raise ValueError(f"Unsupported trace format: {trace_path}")


def summarise_steps(steps: Iterable[Mapping[str, object]]) -> Dict[str, float]:
    total = 0
    edits = 0
    test_runs = 0
    test_failures = 0
    diagnostics = 0
    foreground = 0

    for step in steps:
        total += 1
        action = str(step.get("action", ""))
        if action in {"edit", "apply_patch"}:
            edits += 1
        if action == "run_tests":
            test_runs += 1
            metadata = step.get("metadata")
            if isinstance(metadata, Mapping):
                fails = metadata.get("failures") or []
                if isinstance(fails, list) and fails:
                    test_failures += 1
        if action in {"run_lint", "compile"}:
            diagnostics += 1
        if action == "plan_update":
            summary = step.get("metadata", {}).get("summary") if isinstance(step.get("metadata"), Mapping) else ""
            if isinstance(summary, str) and "STM" in summary:
                foreground += 1

    return {
        "steps": total,
        "edits": edits,
        "test_runs": test_runs,
        "test_failures": test_failures,
        "diagnostics": diagnostics,
        "stm_foreground": foreground,
    }


def run_task(task: str, variant: str) -> Dict[str, object]:
    trace_path = TASK_ROOT / task / f"{variant}.jsonl"
    if not trace_path.exists():
        raise FileNotFoundError(f"Missing trace: {trace_path}")

    adapter = CodeTraceAdapter()
    output_dir = OUTPUT_ROOT / task / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter.run(trace_path, output_dir)

    steps = load_steps(trace_path)
    summary = summarise_steps(steps)

    manifest = {
        "task": task,
        "variant": variant,
        "summary": summary,
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline vs STM coding demos")
    parser.add_argument("--task", choices=[p.name for p in TASK_ROOT.iterdir() if p.is_dir()], help="Limit to a single task", default=None)
    parser.add_argument("--variant", choices=VARIANTS, help="Limit to baseline or STM", default=None)
    args = parser.parse_args()

    tasks = [args.task] if args.task else [p.name for p in sorted(TASK_ROOT.iterdir()) if p.is_dir()]
    variants = [args.variant] if args.variant else list(VARIANTS)

    results: List[Dict[str, object]] = []
    for task in tasks:
        for variant in variants:
            manifest = run_task(task, variant)
            results.append(manifest)

    header = f"{'Task':<28} {'Variant':<9} {'Steps':>5} {'Edits':>5} {'Test Runs':>9} {'Test Fails':>10} {'Diagnostics':>12} {'STM Foreground':>15}"
    print(header)
    print("-" * len(header))
    for result in results:
        task = result["task"]
        variant = result["variant"]
        summary = result["summary"]
        print(
            f"{task:<28} {variant:<9} {summary['steps']:>5} {summary['edits']:>5} {summary['test_runs']:>9} {summary['test_failures']:>10} {summary['diagnostics']:>12} {summary['stm_foreground']:>15}"
        )


if __name__ == "__main__":
    main()
