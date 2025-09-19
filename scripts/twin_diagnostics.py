#!/usr/bin/env python3
"""Summarise ANN twin matches with distance and signature diagnostics."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


def load_twins(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Twin file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarise_result(result: Dict[str, Any]) -> Dict[str, Any]:
    matches: List[Dict[str, Any]] = result.get("matches", [])
    if not matches:
        return {
            "string": result.get("string"),
            "aligned_windows": 0,
            "mean_ann_distance": None,
            "min_ann_distance": None,
            "max_ann_distance": None,
            "top_signatures": [],
        }
    distances = [m.get("distance") for m in matches if "distance" in m]
    signatures = [m.get("signature") for m in matches if m.get("signature")]
    signature_counts = Counter(signatures)
    return {
        "string": result.get("string"),
        "aligned_windows": len(matches),
        "mean_ann_distance": mean(distances) if distances else None,
        "min_ann_distance": min(distances) if distances else None,
        "max_ann_distance": max(distances) if distances else None,
        "top_signatures": signature_counts.most_common(5),
    }


def diagnostics(payload: Dict[str, Any], min_windows: int = 30) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = payload.get("results", payload if isinstance(payload, list) else [])
    filtered = [r for r in results if len(r.get("matches", [])) >= min_windows]

    summaries = [summarise_result(r) for r in filtered]
    total_aligned = sum(item["aligned_windows"] for item in summaries)
    all_distances: List[float] = []
    signature_counter: Counter[str] = Counter()
    for r, summary in zip(filtered, summaries):
        matches = r.get("matches", [])
        all_distances.extend([m.get("distance") for m in matches if "distance" in m])
        signature_counter.update([m.get("signature") for m in matches if m.get("signature")])
    aggregate = {
        "num_strings": len(summaries),
        "total_aligned_windows": total_aligned,
        "overall_mean_ann_distance": mean(all_distances) if all_distances else None,
        "top_signature_tokens": signature_counter.most_common(10),
        "strings": summaries,
    }
    return aggregate


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python scripts/twin_diagnostics.py <twins.json> [min_windows]")
    twin_path = Path(sys.argv[1])
    min_windows = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    payload = load_twins(twin_path)
    report = diagnostics(payload, min_windows=min_windows)
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
