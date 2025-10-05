#!/usr/bin/env python3
"""Evaluate reliability checkpoint under structured sparsity settings."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.reality_filter_eval import ReliabilityModelWrapper

SUPPORTED_LABEL = "SUPPORTED"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detail_path", type=Path, help="eval_detail JSONL file to evaluate")
    parser.add_argument("checkpoint", type=Path, help="Reliability checkpoint (.pt)")
    parser.add_argument("--local-window", nargs="*", type=int, default=[32, 64])
    parser.add_argument("--rupture-tokens", nargs="*", type=int, default=[4, 8])
    parser.add_argument("--admit-threshold", type=float, default=0.2)
    parser.add_argument("--margin-threshold", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_records(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def select_evidence(record: Dict[str, object], local_window: int, rupture_tokens: int) -> List[Dict[str, object]]:
    sentences = record.get("sentences") or []
    if not isinstance(sentences, list):
        return []
    local_slice = sentences[:local_window]
    remainder = sentences[local_window:]
    sortable: List[Tuple[float, Dict[str, object]]] = []
    for item in remainder:
        if not isinstance(item, dict):
            continue
        metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
        hazard = float(metrics.get("lambda", 0.0))
        sortable.append((hazard, item))
    sortable.sort(key=lambda pair: pair[0], reverse=True)
    rupture_slice = [entry for _, entry in sortable[:rupture_tokens]]
    selected = local_slice + rupture_slice
    evidence_payload: List[Dict[str, object]] = []
    for entry in selected:
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("sentence") or "").strip()
        metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
        evidence_payload.append({"text": text, "metrics": metrics})
    return evidence_payload


def evaluate_setting(
    records: Iterable[Dict[str, object]],
    model: ReliabilityModelWrapper,
    *,
    local_window: int,
    rupture_tokens: int,
) -> Dict[str, float]:
    tp = fp = fn = tn = 0
    probs: List[float] = []
    margins: List[float] = []
    labels: List[int] = []
    evidence_counts: List[int] = []

    start = time.perf_counter()
    for record in records:
        question = str(record.get("question") or "")
        final_answer = str(record.get("final_answer") or "")
        baseline_answer = str(record.get("baseline_answer") or "")
        evidence = select_evidence(record, local_window, rupture_tokens)
        evidence_counts.append(len(evidence))
        prob, margin = model.score(
            question=question,
            candidate=final_answer,
            baseline=baseline_answer,
            evidence=evidence,
        )
        predicted = model.should_admit(prob, margin)
        label = 1 if str(record.get("expected")).upper() == SUPPORTED_LABEL else 0
        if predicted and label:
            tp += 1
        elif predicted and not label:
            fp += 1
        elif (not predicted) and label:
            fn += 1
        else:
            tn += 1
        probs.append(prob)
        margins.append(margin)
        labels.append(label)
    elapsed = time.perf_counter() - start
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    records_per_sec = len(probs) / elapsed if elapsed > 0 else float("inf")
    avg_evidence = mean(evidence_counts) if evidence_counts else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "records": len(probs),
        "avg_evidence": avg_evidence,
        "records_per_second": records_per_sec,
    }


def main() -> None:
    args = parse_args()
    records = load_records(args.detail_path)
    max_evidence = max((lw + rt) for lw in args.local_window for rt in args.rupture_tokens)
    model = ReliabilityModelWrapper(
        args.checkpoint,
        device=args.device,
        admit_threshold=args.admit_threshold,
        margin_threshold=args.margin_threshold,
        max_evidence=max_evidence,
    )

    results: Dict[str, Dict[str, float]] = {}
    for lw in args.local_window:
        for rt in args.rupture_tokens:
            key = f"local_{lw}_rupture_{rt}"
            metrics = evaluate_setting(records, model, local_window=lw, rupture_tokens=rt)
            results[key] = metrics
            print(json.dumps({"setting": key, **metrics}, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "detail_path": str(args.detail_path),
        "checkpoint": str(args.checkpoint),
        "settings": results,
        "admit_threshold": args.admit_threshold,
        "margin_threshold": args.margin_threshold,
    }, indent=2))


if __name__ == "__main__":
    main()
