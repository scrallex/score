#!/usr/bin/env python3
"""Generate LaTeX tables for the whitepaper receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple


def load_summary(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "summary" not in data:
        raise ValueError(f"Summary section missing in {path}")
    return data["summary"]


def format_float(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def write_table(table_path: Path, header: str, rows: List[str]) -> None:
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", encoding="utf-8") as fh:
        fh.write("\\begin{tabular}{" + header + "}\n")
        fh.write("\\toprule\n")
        for row in rows:
            fh.write(row + "\\\\\n")
        fh.write("\\bottomrule\n")
        fh.write("\\end{tabular}\n")


def build_stm_table(out_path: Path, entries: List[Tuple[str, Path]]) -> None:
    """Create Table A: STM Logistics guardrail summary."""
    lines: List[str] = ["Configuration & Coverage (\%) & Lead (steps) & $p_{min}$ & Source"]
    for label, summary_path in entries:
        summary = load_summary(summary_path)
        coverage = float(summary.get("coverage_weighted", 0.0)) * 100
        lead = summary.get("lead_mean") or 0.0
        p_min = summary.get("p_value_min") or 0.0
        lines.append(
            f"{label} & {format_float(coverage, 2)} & {format_float(float(lead or 0.0), 2)} & {format_float(float(p_min or 0.0), 3)} & {summary_path.as_posix()}"
        )

    write_table(out_path, "lcccc", lines)


def build_spt_table(out_path: Path, receipts: List[Tuple[str, str]]) -> None:
    lines = ["Evidence & Location"]
    for label, ref in receipts:
        lines.append(f"{label} & {ref}")
    write_table(out_path, "ll", lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX tables for STM↔spt receipts")
    parser.add_argument("--stm", nargs="+", type=str, help="Pairs label=path to permutation summaries", required=True)
    parser.add_argument("--spt", nargs="+", type=str, help="Pairs label=reference path", required=True)
    parser.add_argument("--stm-out", type=Path, default=Path("score/docs/whitepaper/table_stm_guardrail.tex"))
    parser.add_argument("--spt-out", type=Path, default=Path("score/docs/whitepaper/table_spt_receipts.tex"))
    args = parser.parse_args()

    stm_entries: List[Tuple[str, Path]] = []
    for pair in args.stm:
        if "=" not in pair:
            raise ValueError(f"Expected label=path for --stm entries (got {pair})")
        label, path_str = pair.split("=", 1)
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(path)
        stm_entries.append((label, path))

    spt_entries: List[Tuple[str, str]] = []
    for pair in args.spt:
        if "=" not in pair:
            raise ValueError(f"Expected label=reference for --spt entries (got {pair})")
        label, ref = pair.split("=", 1)
        spt_entries.append((label, ref))

    build_stm_table(args.stm_out, stm_entries)
    build_spt_table(args.spt_out, spt_entries)


if __name__ == "__main__":
    main()
