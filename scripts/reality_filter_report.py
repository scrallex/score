#!/usr/bin/env python3
"""Assemble report artifacts for one or more truth-packs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packs", nargs="+", required=True, help="Pack identifiers (e.g., docs_demo)")
    parser.add_argument("--output-dir", type=Path, default=Path("results/report"))
    return parser.parse_args()


def load_metrics(pack: str) -> Dict[str, object]:
    metrics_candidates = [
        Path(f"results/{pack}_metrics.json"),
        Path("results/semantic_guardrail_metrics.json"),
    ]
    for path in metrics_candidates:
        if path.exists():
            return json.loads(path.read_text())
    return {}


def load_permutation(pack: str) -> Dict[str, object]:
    path = Path(f"results/permutation/{pack}.json")
    return json.loads(path.read_text()) if path.exists() else {}


def load_sweep(pack: str) -> List[Dict[str, str]]:
    path = Path(f"results/sweeps/{pack}.csv")
    if not path.exists():
        return []
    rows: List[Dict[str, str]] = []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        rows.extend(reader)
    return rows


def load_eval_summary(pack: str) -> Dict[str, object]:
    path = Path(f"results/eval/{pack}/eval_summary.json")
    return json.loads(path.read_text()) if path.exists() else {}


def load_eval_detail(pack: str) -> List[Dict[str, object]]:
    path = Path(f"results/eval/{pack}/eval_detail.jsonl")
    if not path.exists():
        return []
    records: List[Dict[str, object]] = []
    with path.open() as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    return records


def _build_confusion_from_detail(records: Iterable[Mapping[str, object]]) -> Dict[str, Dict[str, int]]:
    confusion: Dict[str, Dict[str, int]] = {}
    for record in records:
        expected = record.get("expected")
        predicted = record.get("predicted")
        if expected is None or predicted is None:
            continue
        row = confusion.setdefault(str(expected), {})
        row[str(predicted)] = row.get(str(predicted), 0) + 1
    return confusion


def validate_eval_parity(eval_summary: Dict[str, object], detail_records: List[Dict[str, object]]) -> None:
    if not eval_summary or not detail_records:
        return

    metrics = eval_summary.get("metrics", {})
    if isinstance(metrics, dict) and "total" in metrics:
        summary_total = int(metrics["total"])
        if summary_total != len(detail_records):
            raise ValueError(
                f"Evaluation summary total ({summary_total}) does not match detail count ({len(detail_records)})."
            )

    summary_confusion = eval_summary.get("confusion_matrix")
    if isinstance(summary_confusion, dict) and summary_confusion:
        detail_confusion = _build_confusion_from_detail(detail_records)
        for expected, summary_row in summary_confusion.items():
            detail_row = detail_confusion.get(expected, {})
            for predicted, summary_count in summary_row.items():
                detail_count = int(detail_row.get(predicted, 0))
                if int(summary_count) != detail_count:
                    raise ValueError(
                        f"Confusion mismatch for ({expected}, {predicted}): summary={summary_count}, detail={detail_count}."
                    )
            # No unexpected predictions allowed in detail
            extras = set(detail_row) - set(summary_row)
            if extras:
                raise ValueError(f"Detail confusion contains unexpected predictions for {expected}: {sorted(extras)}")
        # Ensure no extra expected classes in detail
        extra_expected = set(detail_confusion) - set(summary_confusion)
        if extra_expected:
            raise ValueError(f"Detail confusion contains unexpected expected labels: {sorted(extra_expected)}")


def summarise(pack: str, output_dir: Path) -> None:
    metrics = load_metrics(pack)
    permutation = load_permutation(pack)
    sweep_rows = load_sweep(pack)
    eval_summary = load_eval_summary(pack)
    detail_records = load_eval_detail(pack)

    validate_eval_parity(eval_summary, detail_records)

    eval_examples = detail_records[:10]

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{pack}.md"

    with summary_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# Reality Filter Report — {pack}\n\n")
        if metrics:
            fh.write("## KPIs\n\n")
            for key, value in metrics.get("kpis", {}).items():
                fh.write(f"- **{key}**: {value}\n")
            fh.write("\n")
        if permutation:
            fh.write("## Permutation Test\n\n")
            fh.write(
                f"Observed coverage {permutation.get('observed_coverage', 0):.3f} with p-value {permutation.get('p_value', 1.0):.4f}.\n\n"
            )
        if sweep_rows:
            fh.write("## Threshold Sweep (top coverage)\n\n")
            best = max(sweep_rows, key=lambda r: float(r.get("coverage", 0.0)))
            fh.write(
                "| r_min | lambda_max | sigma_min | coverage | admitted | total |\n"
                "| --- | --- | --- | --- | --- | --- |\n"
            )
            fh.write(
                f"| {best['r_min']} | {best['lambda_max']} | {best['sigma_min']} | {float(best['coverage']):.3f} "
                f"| {best['admitted']} | {best['total']} |\n\n"
            )
            fh.write(f"Full CSV: results/sweeps/{pack}.csv\n\n")
        if eval_summary:
            fh.write("## Evaluation Metrics\n\n")
            for key in [
                "macro_f1",
                "baseline_macro_f1",
                "macro_f1_delta",
                "dev_macro_f1",
                "sanity_flags_count",
            ]:
                if key in eval_summary:
                    fh.write(f"- **{key}**: {eval_summary[key]}\n")
            metrics = eval_summary.get("metrics", {})
            if isinstance(metrics, dict):
                for m_key, m_value in metrics.items():
                    fh.write(f"- **metrics.{m_key}**: {m_value}\n")
            best = eval_summary.get("best_thresholds", {})
            if isinstance(best, dict):
                fh.write("- **best_thresholds**:\n")
                for b_key, b_value in best.items():
                    fh.write(f"  - {b_key}: {b_value}\n")
            fh.write("\n")
        if eval_examples:
            fh.write("## Before vs After Examples\n\n")
            for example in eval_examples:
                fh.write(f"### {example.get('id')} ({example.get('expected')} → {example.get('predicted')})\n\n")
                fh.write("**Question:** " + str(example.get("question", "")) + "\n\n")
                baseline_answer = example.get("baseline_answer") or example.get("raw_answer") or "—"
                fh.write("**Baseline:** " + str(baseline_answer) + "\n\n")
                fh.write("**Filtered:** " + str(example.get("final_answer", "")) + "\n\n")
                fh.write("---\n")
        fh.write("---\nGenerated by scripts/reality_filter_report.py\n")

    print(f"Report written to {summary_path}")


def main() -> None:
    args = parse_args()
    for pack in args.packs:
        summarise(pack, args.output_dir)


if __name__ == "__main__":
    main()
