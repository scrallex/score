#!/usr/bin/env python3
"""Generate lead-time outputs for multiple onset hypotheses."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch lead-time sweep")
    parser.add_argument("--state", required=True, help="State JSON path")
    parser.add_argument("--times", nargs="+", required=True, help="Onset times (HH:MM)")
    parser.add_argument("--output", type=Path, default=Path("docs/note/tab4a_midnight_onset_sweep.csv"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("analysis"))
    parser.add_argument("--plots-dir", type=Path, default=Path("docs/plots"))
    parser.add_argument("--date", default="2017-09-08", help="Onset date (YYYY-MM-DD)")
    return parser.parse_args()


def run_cli_lead(state: str, onset_iso: str, json_path: Path, plot_path: Path) -> dict:
    cmd = [
        "python",
        "-m",
        "sep_text_manifold.cli_lead",
        "--state",
        state,
        "--onset",
        onset_iso,
        "--output",
        str(json_path),
        "--plot",
        str(plot_path),
    ]
    subprocess.run(cmd, check=True)
    return json.loads(json_path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    rows = []
    state_path = Path(args.state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for time_str in args.times:
        onset_iso = f"{args.date}T{time_str}:00"
        suffix = time_str.replace(":", "")
        json_path = args.analysis_dir / f"mms_0000_lead_{suffix}.json"
        plot_path = args.plots_dir / f"mms_0000_lead_{suffix}.png"
        payload = run_cli_lead(str(state_path), onset_iso, json_path, plot_path)
        bins = payload.get("bins", [])
        last_density = bins[-1].get("density", 0.0) if bins else 0.0
        rows.append(
            {
                "onset": onset_iso,
                "fg_density_last_bin": last_density,
                "monotonic_increase": payload.get("monotonic_increase"),
            }
        )
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=["onset", "fg_density_last_bin", "monotonic_increase"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
