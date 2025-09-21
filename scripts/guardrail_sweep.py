#!/usr/bin/env python3
"""Sweep STM guardrail targets and report coverage/p-value statistics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, Sequence

from calibrate_router import load_state, extract_metrics, compute_configuration
from run_permutation_guardrail import summarise_domain


def sweep_guardrails(
    *,
    state_path: Path,
    domain_root: Path,
    iterations: int,
    targets: Sequence[float],
    coverage_window: float,
    output_dir: Path,
    prefix: str,
) -> list[dict[str, float | int | None]]:
    metrics = extract_metrics(load_state(state_path))
    rows: list[dict[str, float | int | None]] = []

    for target in targets:
        lower = max(target - coverage_window, 0.001)
        upper = target + coverage_window
        cfg, _, coverage = compute_configuration(metrics, lower, upper)

        pct = int(round(target * 100))
        cfg_path = output_dir / f"router_config_{prefix}_{pct:02d}pct.json"
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        summary = summarise_domain(
            domain_root=domain_root,
            config_path=cfg_path,
            output_path=cfg_path.with_suffix(".permutation.json"),
            iterations=iterations,
        )

        coverage_weighted = float(summary.get("coverage_weighted", 0.0)) * 100
        coverage_mean = float(summary.get("coverage_mean", 0.0)) * 100
        p_mean = float(summary.get("p_value_mean") or 0.0)
        p_min = float(summary.get("p_value_min") or 0.0)
        p_ci = summary.get("p_value_ci95") or (None, None)
        lead_mean = float(summary.get("lead_mean") or 0.0)
        lead_ci = summary.get("lead_ci95") or (None, None)
        precision_mean = summary.get("precision_mean")

        rows.append(
            {
                "target_guardrail": float(target),
                "actual_coverage_pct": coverage_mean,
                "coverage_weighted_pct": coverage_weighted,
                "lead_mean": lead_mean,
                "lead_ci_low": float(lead_ci[0]) if lead_ci[0] is not None else None,
                "lead_ci_high": float(lead_ci[1]) if lead_ci[1] is not None else None,
                "lead_hits": int(summary.get("lead_hits", 0) or 0),
                "p_value_mean": p_mean,
                "p_value_min": p_min,
                "p_value_ci_low": float(p_ci[0]) if p_ci and p_ci[0] is not None else None,
                "p_value_ci_high": float(p_ci[1]) if p_ci and p_ci[1] is not None else None,
                "precision_mean": float(precision_mean) if precision_mean is not None else None,
                "trace_count": int(summary.get("trace_count", 0) or 0),
            }
        )

    return rows


def write_appendix(
    *,
    csv_path: Path,
    domain_label: str,
    rows: Iterable[dict[str, float | int | None]],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        if not file_exists:
            writer.writerow(
                [
                    "domain",
                    "target_guardrail",
                    "actual_coverage_pct",
                    "coverage_weighted_pct",
                    "lead_mean",
                    "lead_ci_low",
                    "lead_ci_high",
                    "lead_hits",
                    "p_value_mean",
                    "p_value_min",
                    "p_value_ci_low",
                    "p_value_ci_high",
                    "precision_mean",
                    "trace_count",
                ]
            )
        for row in rows:
            def fmt(value: float | int | None) -> str:
                if value is None:
                    return ""
                if isinstance(value, int):
                    return str(value)
                return f"{value:.3f}"

            writer.writerow(
                [
                    domain_label,
                    fmt(row.get("target_guardrail")),
                    fmt(row.get("actual_coverage_pct")),
                    fmt(row.get("coverage_weighted_pct")),
                    fmt(row.get("lead_mean")),
                    fmt(row.get("lead_ci_low")),
                    fmt(row.get("lead_ci_high")),
                    fmt(row.get("lead_hits")),
                    fmt(row.get("p_value_mean")),
                    fmt(row.get("p_value_min")),
                    fmt(row.get("p_value_ci_low")),
                    fmt(row.get("p_value_ci_high")),
                    fmt(row.get("precision_mean")),
                    fmt(row.get("trace_count")),
                ]
            )


def write_summary_json(
    *,
    json_path: Path,
    metadata: dict[str, object],
    rows: Iterable[dict[str, float | int | None]],
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "rows": list(rows),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_target_grid(base_targets: Sequence[float], grid: Sequence[float] | None) -> list[float]:
    targets = list(base_targets)
    if grid:
        start, stop, step = grid
        if step <= 0:
            raise ValueError("Grid step must be greater than zero")
        if stop < start:
            raise ValueError("Grid stop must be >= start")
        count = int(round((stop - start) / step))
        grid_targets = [round(start + idx * step, 6) for idx in range(count + 1)]
        if grid_targets[-1] < stop - 1e-9:
            grid_targets.append(round(stop, 6))
        targets.extend(grid_targets)
    deduped = sorted({round(t, 6) for t in targets})
    return [t for t in deduped if t > 0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Guardrail sweep across coverage targets")
    parser.add_argument("state", type=Path, help="STM state JSON (invalid traces)")
    parser.add_argument("domain_root", type=Path, help="Domain root containing invalid/metrics")
    parser.add_argument("--prefix", default="sweep", help="Filename prefix for generated configs")
    parser.add_argument(
        "--targets",
        nargs="*",
        type=float,
        default=None,
        help="Coverage targets (fractions)",
    )
    parser.add_argument(
        "--grid",
        nargs=3,
        type=float,
        metavar=("START", "STOP", "STEP"),
        help="Generate coverage targets from START to STOP inclusive using STEP increments",
    )
    parser.add_argument("--iteration", dest="iterations", type=int, default=20000, help="Permutation iterations")
    parser.add_argument("--window", type=float, default=0.005, help="Coverage window for calibration")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis"),
        help="Directory to store generated configs",
    )
    parser.add_argument(
        "--appendix",
        type=Path,
        default=Path("docs/note/appendix_guardrail_sweep.csv"),
        help="CSV file to append sweep results",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Write sweep metadata/results to this JSON file (defaults to <output-dir>/guardrail_sweep_<prefix>.summary.json)",
    )
    parser.add_argument("--label", type=str, default="PlanBench", help="Label for appendix rows")
    args = parser.parse_args()

    default_targets = [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    base_targets = args.targets if args.targets else default_targets
    targets = build_target_grid(base_targets, args.grid)

    rows = sweep_guardrails(
        state_path=args.state,
        domain_root=args.domain_root,
        iterations=args.iterations,
        targets=targets,
        coverage_window=args.window,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )
    summary_path = args.summary_json
    if summary_path is None:
        summary_path = args.output_dir / f"guardrail_sweep_{args.prefix}.summary.json"

    metadata = {
        "state": str(args.state),
        "domain_root": str(args.domain_root),
        "iterations": args.iterations,
        "coverage_window": args.window,
        "prefix": args.prefix,
        "label": args.label,
        "targets": targets,
    }

    write_summary_json(json_path=summary_path, metadata=metadata, rows=rows)
    write_appendix(csv_path=args.appendix, domain_label=args.label, rows=rows)


if __name__ == "__main__":
    main()
