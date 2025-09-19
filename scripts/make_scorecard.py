#!/usr/bin/env python3
"""Build scorecard CSV combining coverage, diagnostics, and lead-time stats."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
OUT_DIR = ROOT / "docs" / "note"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SliceConfig = Dict[str, str]

SLICES: List[SliceConfig] = [
    {
        "label": "2017-09-07 22:30–23:30",
        "slug": "2230",
        "coverage": "router_config_2017-09-07_2230-2330.coverage.json",
        "proposals": "mms_proposals_struct_2017-09-07_2230-2330.json",
        "diagnostics": "mms_twins_2230_diagnostics.json",
        "lead": "mms_2230_leadtime.json",
    },
    {
        "label": "2017-09-07 23:00–00:00",
        "slug": "2300",
        "coverage": "router_config_2017-09-07_2300-0000.coverage.json",
        "proposals": "mms_proposals_struct_2300.json",
        "diagnostics": "mms_twins_2300_diagnostics.json",
        "lead": "mms_2300_leadtime.json",
    },
    {
        "label": "2017-09-08 00:00–01:00",
        "slug": "0000",
        "coverage": "router_config_0000.coverage.json",
        "proposals": "mms_0000_proposals_struct.json",
        "diagnostics": "mms_twins_0000_diagnostics.json",
        "lead": "mms_0000_leadtime.json",
    },
    {
        "label": "2017-09-08 01:00–02:00",
        "slug": "0100",
        "coverage": "router_config_0100.coverage.json",
        "proposals": "mms_0100_proposals_struct.json",
        "diagnostics": "mms_twins_0100_diagnostics.json",
        "lead": "mms_0100_leadtime.json",
    },
    {
        "label": "2017-09-10 00:00–01:00 (quiet)",
        "slug": "quiet",
        "coverage": "router_config_quiet.coverage.json",
        "proposals": "mms_quiet_proposals_struct.json",
        "diagnostics": "mms_twins_quiet_diagnostics.json",
        "lead": "mms_quiet_leadtime.json",
    },
]


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def lead_last_density(lead_payload: dict) -> float:
    bins = lead_payload.get("bins", [])
    if not bins:
        return 0.0
    return float(bins[-1].get("density", 0.0))


def summarise_slice(cfg: SliceConfig) -> dict:
    coverage_path = ANALYSIS / cfg["coverage"]
    diag_path = ANALYSIS / cfg["diagnostics"]
    lead_path = ANALYSIS / cfg["lead"]
    proposals_path = ANALYSIS / cfg["proposals"]

    coverage = load_json(coverage_path).get("coverage", 0.0)
    diagnostics = load_json(diag_path)
    lead = load_json(lead_path)
    proposals = load_json(proposals_path).get("proposals", [])

    top_qgrams = diagnostics.get("top_signature_tokens", [])
    top_qgrams_str = " | ".join(q for q, _ in top_qgrams[:3]) if top_qgrams else ""
    mean_ann = diagnostics.get("overall_mean_ann_distance")

    row = {
        "slice": cfg["label"],
        "fg_coverage_percent": round(coverage * 100, 2),
        "proposals_count": len(proposals),
        "twins_ge_50": diagnostics.get("num_strings", 0),
        "aligned_windows_total": diagnostics.get("total_aligned_windows", 0),
        "mean_ann_distance": round(mean_ann, 6) if isinstance(mean_ann, (int, float)) and mean_ann is not None else "",
        "lead_density_percent_minus5": round(lead_last_density(lead) * 100, 2),
        "top_signature_tokens": top_qgrams_str,
    }
    return row


def main() -> None:
    rows = [summarise_slice(cfg) for cfg in SLICES]
    output_csv = OUT_DIR / "tab1_scorecard.csv"
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
