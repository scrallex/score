#!/usr/bin/env python3
"""Aggregate STM PlanBench experiment outputs into a CSV table."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def infer_domain(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith("bw_") or "blocksworld" in lowered:
        return "Blocksworld"
    if lowered.startswith("mystery"):
        return "Mystery Blocksworld"
    if lowered.startswith("logistics"):
        return "Logistics"
    return name


def aggregate(root: Path) -> List[Dict[str, object]]:
    rows: Dict[str, Dict[str, object]] = {}
    for summary_path in root.rglob("summary.json"):
        if "metrics" not in summary_path.parts:
            continue
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        lead_records = data.get("lead_records", [])
        twin_records = {rec.get("trace"): rec for rec in data.get("twin_records", [])}
        for rec in lead_records:
            trace = rec.get("trace", "")
            domain = infer_domain(trace)
            row = rows.setdefault(
                domain,
                {
                    "domain": domain,
                    "traces": 0,
                    "alerts_total": 0,
                    "failures_total": 0,
                    "lead_coverage_sum": 0.0,
                    "lead_mean_sum": 0.0,
                    "lead_max": float("-inf"),
                    "lead_min": float("inf"),
                    "twin_corrected": 0,
                    "twin_total": 0,
                },
            )
            stats = rec.get("stats", {})
            coverage = float(stats.get("coverage", 0.0))
            leads = stats.get("leads", []) or []
            row["traces"] += 1
            row["alerts_total"] += len(rec.get("alerts", []))
            row["failures_total"] += len(rec.get("val_failures", []))
            row["lead_coverage_sum"] += coverage
            row["lead_mean_sum"] += float(stats.get("mean", 0.0))
            if leads:
                row["lead_max"] = max(row["lead_max"], max(leads))
                row["lead_min"] = min(row["lead_min"], min(leads))
            twin = twin_records.get(trace, {})
            row["twin_corrected"] += int(twin.get("corrected", 0))
            row["twin_total"] += int(twin.get("total_failures", 0))
    results: List[Dict[str, object]] = []
    for row in rows.values():
        traces = max(1, row["traces"])
        lead_max = row["lead_max"] if row["lead_max"] != float("-inf") else 0
        lead_min = row["lead_min"] if row["lead_min"] != float("inf") else 0
        twin_total = max(1, row["twin_total"])
        results.append(
            {
                "domain": row["domain"],
                "traces": row["traces"],
                "alerts_per_trace": row["alerts_total"] / traces,
                "failures_per_trace": row["failures_total"] / traces,
                "lead_coverage": row["lead_coverage_sum"] / traces,
                "lead_mean": row["lead_mean_sum"] / traces,
                "lead_max": lead_max,
                "lead_min": lead_min,
                "twin_correction_rate": row["twin_corrected"] / twin_total,
            }
        )
    results.sort(key=lambda item: item["domain"])
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate STM PlanBench metrics")
    parser.add_argument("--input-root", type=Path, default=Path("output/planbench_public"))
    parser.add_argument("--output", type=Path, default=Path("docs/note/planbench_scorecard.csv"))
    args = parser.parse_args()

    rows = aggregate(args.input_root)
    if not rows:
        raise SystemExit(f"No summary.json files found under {args.input_root}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
