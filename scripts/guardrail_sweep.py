#!/usr/bin/env python3
"""Sweep STM guardrail targets and report coverage/p-value statistics."""

from __future__ import annotations

import argparse
import csv
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
) -> list[dict[str, float | str]]:
    metrics = extract_metrics(load_state(state_path))
    rows: list[dict[str, float | str]] = []

    for target in targets:
        lower = max(target - coverage_window, 0.001)
        upper = target + coverage_window
        cfg, _, coverage = compute_configuration(metrics, lower, upper)

        pct = int(round(target * 100))
        cfg_path = output_dir / f"router_config_{prefix}_{pct:02d}pct.json"
        cfg_path.write_text(__import__("json").dumps(cfg, indent=2), encoding="utf-8")

        summary = summarise_domain(
            domain_root=domain_root,
            config_path=cfg_path,
            output_path=cfg_path.with_suffix(".permutation.json"),
            iterations=iterations,
        )

        coverage_mean = float(summary.get("coverage_weighted", 0.0)) * 100
        p_mean = float(summary.get("p_value_mean") or 0.0)
        p_min = float(summary.get("p_value_min") or 0.0)
        lead_mean = float(summary.get("lead_mean") or 0.0)

        rows.append(
            {
                "target_guardrail": f"{target:.3f}",
                "actual_coverage_pct": f"{coverage_mean:.3f}",
                "lead_mean": f"{lead_mean:.3f}",
                "p_value_mean": f"{p_mean:.3f}",
                "p_value_min": f"{p_min:.3f}",
            }
        )

    return rows


def write_appendix(
    *,
    csv_path: Path,
    domain_label: str,
    rows: Iterable[dict[str, float | str]],
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
                    "lead_mean",
                    "p_value_mean",
                    "p_value_min",
                ]
            )
        for row in rows:
            writer.writerow(
                [
                    domain_label,
                    row["target_guardrail"],
                    row["actual_coverage_pct"],
                    row["lead_mean"],
                    row["p_value_mean"],
                    row["p_value_min"],
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Guardrail sweep across coverage targets")
    parser.add_argument("state", type=Path, help="STM state JSON (invalid traces)")
    parser.add_argument("domain_root", type=Path, help="Domain root containing invalid/metrics")
    parser.add_argument("--prefix", default="sweep", help="Filename prefix for generated configs")
    parser.add_argument(
        "--targets",
        nargs="*",
        type=float,
        default=[0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
        help="Coverage targets (fractions)",
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
    parser.add_argument("--label", type=str, default="PlanBench", help="Label for appendix rows")
    args = parser.parse_args()

    rows = sweep_guardrails(
        state_path=args.state,
        domain_root=args.domain_root,
        iterations=args.iterations,
        targets=args.targets,
        coverage_window=args.window,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )
    write_appendix(csv_path=args.appendix, domain_label=args.label, rows=rows)


if __name__ == "__main__":
    main()
