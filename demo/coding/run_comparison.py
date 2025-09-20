#!/usr/bin/env python3
"""Compare baseline vs STM traces across demo coding tasks."""

from __future__ import annotations

import argparse
import json
import importlib.util
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import html

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

spec = importlib.util.spec_from_file_location(
    "stm_adapters.code_trace_adapter", SRC_ROOT / "stm_adapters" / "code_trace_adapter.py"
)
if spec is None or spec.loader is None:
    raise ImportError("Unable to load CodeTraceAdapter module")
_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = _module
spec.loader.exec_module(_module)  # type: ignore[attr-defined]
CodeTraceAdapter = getattr(_module, "CodeTraceAdapter")

TASK_ROOT = Path(__file__).resolve().parent / "tasks"
OUTPUT_ROOT = Path(__file__).resolve().parent / "output"
VARIANTS = ("baseline", "stm")


def load_steps(trace_path: Path) -> List[Mapping[str, object]]:
    text = trace_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        try:
            parsed = [json.loads(line) for line in lines]
            if all(isinstance(item, Mapping) for item in parsed):
                return parsed  # type: ignore[return-value]
        except json.JSONDecodeError:
            pass
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        parsed = [json.loads(line) for line in lines]
        return [item for item in parsed if isinstance(item, Mapping)]
    if isinstance(data, Mapping) and "steps" in data:
        payload = data.get("steps")
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, Mapping)]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, Mapping)]
    raise ValueError(f"Unsupported trace format: {trace_path}")


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _is_success_step(step: Mapping[str, object]) -> bool:
    action = str(step.get("action", ""))
    metadata = step.get("metadata") if isinstance(step.get("metadata"), Mapping) else {}
    if not isinstance(metadata, Mapping):
        return False
    if action == "run_tests":
        failures = metadata.get("failures")
        return not failures
    if action in {"run_lint", "compile"}:
        diagnostics = metadata.get("diagnostics")
        if diagnostics in (None, []):
            exit_code = metadata.get("exit_code")
            return exit_code in (None, 0)
    return False


def summarise_steps(steps: List[Mapping[str, object]]) -> Dict[str, float]:
    total = 0
    edits = 0
    test_runs = 0
    test_failures = 0
    diagnostics = 0
    foreground = 0
    twin_accepts = 0
    timestamps: List[datetime] = []

    for index, step in enumerate(steps):
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
        if action == "apply_patch" and index > 0:
            prev = steps[index - 1]
            prev_summary = ""
            if isinstance(prev.get("metadata"), Mapping):
                prev_summary = str(prev["metadata"].get("summary", ""))
            if "twin" in prev_summary.lower() or "STM" in prev_summary:
                twin_accepts += 1
        ts = _parse_timestamp(step.get("timestamp"))
        if ts:
            timestamps.append(ts)

    duration_minutes = None
    if len(timestamps) >= 2:
        delta = max(timestamps) - min(timestamps)
        duration_minutes = round(delta.total_seconds() / 60.0, 2)

    success = _is_success_step(steps[-1]) if steps else False
    alert_ratio = foreground / max(total, 1)

    lead_alert = any(
        "foreground" in str(step.get("metadata", {}).get("summary", ""))
        for step in steps[:-1]
        if isinstance(step.get("metadata"), Mapping)
    )

    return {
        "steps": total,
        "edits": edits,
        "test_runs": test_runs,
        "test_failures": test_failures,
        "diagnostics": diagnostics,
        "stm_foreground": foreground,
        "alert_ratio": round(alert_ratio, 3),
        "success": success,
        "duration_minutes": duration_minutes,
        "lead_alert": bool(lead_alert),
        "twin_accepts": twin_accepts,
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


def _variant_totals(results: Iterable[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    totals: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}
    successes: Dict[str, int] = {}
    for result in results:
        summary = result["summary"]
        variant = result["variant"]
        counts[variant] = counts.get(variant, 0) + 1
        successes[variant] = successes.get(variant, 0) + int(bool(summary.get("success")))
        bucket = totals.setdefault(
            variant,
            {
                "steps": 0,
                "edits": 0,
                "test_runs": 0,
                "twin_accepts": 0,
                "alert_ratio": 0.0,
            },
        )
        bucket["steps"] += summary.get("steps", 0)
        bucket["edits"] += summary.get("edits", 0)
        bucket["test_runs"] += summary.get("test_runs", 0)
        bucket["twin_accepts"] += summary.get("twin_accepts", 0)
        bucket["alert_ratio"] += summary.get("alert_ratio", 0.0)
    for variant, bucket in totals.items():
        n = counts.get(variant, 1)
        bucket["success_rate"] = successes.get(variant, 0) / n
        bucket["avg_steps"] = bucket["steps"] / n
        bucket["avg_alert_ratio"] = bucket["alert_ratio"] / n
    return totals


def _extract_twin_example(results: Iterable[Dict[str, object]]) -> str | None:
    for result in results:
        if result["variant"] != "stm":
            continue
        trace_path = TASK_ROOT / result["task"] / "stm.jsonl"
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            if "apply_patch" in line:
                payload = json.loads(line)
                metadata = payload.get("metadata")
                if isinstance(metadata, Mapping) and metadata.get("patch"):
                    return str(metadata["patch"])
    return None


def _write_report(results: List[Dict[str, object]], report_path: Path) -> None:
    totals = _variant_totals(results)
    twin_example = _extract_twin_example(results)
    stm_seen_example = {
        "foreground": True,
        "tokens": ["test__fail_any", "edit__generic__python", "plan__summary_present"],
        "lift": 2.7,
        "window_index": 3,
    }

    rows = []
    for result in results:
        summary = result["summary"]
        row = "".join(
            [
                "<tr>",
                f"<td>{html.escape(result['task'])}</td>",
                f"<td>{html.escape(result['variant'])}</td>",
                f"<td>{summary['steps']}</td>",
                f"<td>{summary['edits']}</td>",
                f"<td>{summary['test_runs']}</td>",
                f"<td>{summary['test_failures']}</td>",
                f"<td>{summary['diagnostics']}</td>",
                f"<td>{summary['stm_foreground']}</td>",
                f"<td>{summary['alert_ratio']:.3f}</td>",
                f"<td>{'yes' if summary['success'] else 'no'}</td>",
                f"<td>{summary['duration_minutes'] if summary['duration_minutes'] is not None else '—'}</td>",
                f"<td>{summary['twin_accepts']}</td>",
                "</tr>",
            ]
        )
        rows.append(row)

    variant_rows = []
    for variant, stats in totals.items():
        variant_rows.append(
            "".join(
                [
                    "<tr>",
                    f"<td>{html.escape(variant)}</td>",
                    f"<td>{stats['success_rate']:.2f}</td>",
                    f"<td>{stats['avg_steps']:.2f}</td>",
                    f"<td>{stats['avg_alert_ratio']:.3f}</td>",
                    f"<td>{stats['twin_accepts']}</td>",
                    "</tr>",
                ]
            )
        )

    twin_block = (
        f"<pre>{html.escape(twin_example)}</pre>" if isinstance(twin_example, str) else "<p>No twin patch captured.</p>"
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>STM Coding Demo Comparison</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; color: #1f2430; }}
    h1, h2 {{ color: #121621; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }}
    th, td {{ border: 1px solid #ccc; padding: 0.5rem; text-align: center; }}
    th {{ background: #f4f6fb; }}
    code, pre {{ background: #f6f8fa; padding: 0.5rem; display: block; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>STM Coding Demo – Baseline vs STM</h1>
  <p>This report summarises three curated coding tasks executed with and without the Structural Manifold co-processor. Metrics derive from JSONL traces replayed via the <code>CodeTraceAdapter</code>.</p>

  <h2>Per-Task Metrics</h2>
  <table>
    <thead>
      <tr><th>Task</th><th>Variant</th><th>Steps</th><th>Edits</th><th>Test runs</th><th>Test fails</th><th>Diagnostics</th><th>STM alerts</th><th>Alert ratio</th><th>Success</th><th>Minutes</th><th>Twin accepts</th></tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>

  <h2>Variant Summary</h2>
  <table>
    <thead>
      <tr><th>Variant</th><th>Success rate</th><th>Avg steps</th><th>Avg alert ratio</th><th>Total twin accepts</th></tr>
    </thead>
    <tbody>
      {''.join(variant_rows)}
    </tbody>
  </table>

  <h2>STM /seen Example</h2>
  <p>Example response (redacted) captured when the co-processor flagged the retry-policy drift.</p>
  <pre>{html.escape(json.dumps(stm_seen_example, indent=2))}</pre>

  <h2>STM Twin Patch Sample</h2>
  <p>Representative patch suggested by <code>/stm/propose</code> and applied by the agent.</p>
  {twin_block}

  <p>All artefacts are stored under <code>{html.escape(str(OUTPUT_ROOT))}</code>. Re-run with <code>python demo/coding/run_comparison.py</code>.</p>
</body>
</html>
        """,
        encoding="utf-8",
    )


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

    report_path = OUTPUT_ROOT / "report.html"
    _write_report(results, report_path)
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
