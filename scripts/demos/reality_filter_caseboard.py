#!/usr/bin/env python3
"""Terminal caseboard for reality filter evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table


def load_detail(detail_path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with detail_path.open() as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_summary(summary_path: Path) -> Dict[str, object]:
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    return {}


def build_decision_table(record: Dict[str, object]) -> Table:
    table = Table(title="Decision Log", box=box.SIMPLE_HEAVY)
    table.add_column("Sentence", overflow="fold")
    table.add_column("Action", justify="center")
    table.add_column("Repeat", justify="center")
    table.add_column("Hazard", justify="center")
    table.add_column("Semantic", justify="center")
    table.add_column("Twins", overflow="fold")

    for sentence in record.get("sentences", []):
        decisions = sentence.get("decisions", {})
        twins = sentence.get("twins", [])
        twin_text = ", ".join(twin.get("string", "") for twin in twins) or "—"
        action = sentence.get("action", "emit")
        table.add_row(
            sentence.get("sentence", ""),
            action.upper(),
            "✅" if decisions.get("repeat_ok") else "❌",
            "✅" if decisions.get("hazard_ok") else "❌",
            "✅" if decisions.get("semantic_ok") else "❌",
            twin_text,
        )
    return table


def build_answer_panel(title: str, text: str, subtitle: str = "") -> Panel:
    display = text.strip() or "(empty)"
    return Panel(display, title=title, subtitle=subtitle, padding=(1, 2))


def build_kpi_panel(summary: Dict[str, object]) -> Panel:
    kpis = summary.get("kpis") or summary
    lines = []
    for key in ["hallucination_rate", "repair_yield", "citation_coverage", "latency_ms_p50", "latency_ms_p90"]:
        if key in kpis:
            value = kpis[key]
            if isinstance(value, float):
                value = f"{value:.3f}"
            lines.append(f"{key}: {value}")
    body = "\n".join(lines) if lines else "No KPI data"
    return Panel(body, title="KPIs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detail", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--claim-id", type=str, help="Specific claim id to render")
    args = parser.parse_args()

    records = load_detail(args.detail)
    if not records:
        raise SystemExit("No records in detail file")

    record = records[0]
    if args.claim_id:
        for candidate in records:
            if str(candidate.get("id")) == args.claim_id:
                record = candidate
                break

    summary = load_summary(args.summary) if args.summary else {}

    console = Console()
    layout = Layout()
    layout.split_column(
        Layout(name="top", size=5),
        Layout(name="body")
    )
    layout["body"].split_row(
        Layout(name="left"),
        Layout(name="middle"),
        Layout(name="right"),
    )

    raw_panel = build_answer_panel("Baseline LLM Answer", record.get("raw_answer", ""), subtitle=record.get("id", ""))
    final_panel = build_answer_panel("Reality Filter Answer", record.get("final_answer", ""), subtitle=record.get("predicted", ""))
    decision_table = build_decision_table(record)
    kpi_panel = build_kpi_panel(summary)

    layout["top"].update(kpi_panel)
    layout["left"].update(raw_panel)
    layout["middle"].update(decision_table)
    layout["right"].update(final_panel)

    console.print(layout)


if __name__ == "__main__":
    main()
