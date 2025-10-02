#!/usr/bin/env python3
"""Summarise alert behaviour from the semantic guardrail stream."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

STREAM_PATH = Path("results/semantic_guardrail_stream.jsonl")
OUTPUT_PATH = Path("results/final_guardrail_analysis.json")


def load_events(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Stream file not found: {path}")
    events: list[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {len(events)+1}: {exc}") from exc
    if not events:
        raise ValueError("Stream file is empty")
    return events


def summarise(events: list[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(events)
    semantic_alerts = sum(1 for e in events if e.get("naive_semantic_alert"))
    structural_alerts = sum(1 for e in events if e.get("naive_structural_alert"))
    hybrid_alerts = sum(1 for e in events if e.get("hybrid_guardrail_alert"))

    denom = semantic_alerts + structural_alerts
    if denom:
        fpr_reduction = 1.0 - (hybrid_alerts / denom)
    else:
        fpr_reduction = 1.0

    hybrid_event = next((e for e in events if e.get("hybrid_guardrail_alert")), None)

    return {
        "total_events": total,
        "naive_semantic_alerts": semantic_alerts,
        "naive_structural_alerts": structural_alerts,
        "hybrid_guardrail_alerts": hybrid_alerts,
        "false_positive_reduction_rate": fpr_reduction,
        "hybrid_event": hybrid_event,
    }


def main() -> int:
    events = load_events(STREAM_PATH)
    summary = summarise(events)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
