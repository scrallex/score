#!/usr/bin/env python3
"""Utilities to enforce structural-only filters on STM outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def structural_strings(state_path: Path, output_path: Path, top: int, min_occ: int) -> None:
    state = load_json(state_path)
    scores: Dict[str, Dict[str, Any]] = state.get("string_scores", {})
    rows: List[Dict[str, Any]] = []
    for token, payload in scores.items():
        if "__" not in token:
            continue
        occurrences = int(payload.get("occurrences") or payload.get("occ", 0))
        if occurrences < min_occ:
            continue
        metrics = payload.get("metrics", payload)
        rows.append(
            {
                "token": token,
                "pattern": float(payload.get("patternability") or metrics.get("patternability", 0.0)),
                "connector": float(payload.get("connector", 0.0)),
                "occ": occurrences,
            }
        )
    rows.sort(key=lambda item: (-item["pattern"], -item["connector"], -item["occ"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, item in enumerate(rows[:top], start=1):
            handle.write(
                f"{idx:3d}. {item['token']} pattern={item['pattern']:.3f} connector={item['connector']:.3f} occ={item['occ']}\n"
            )


def structural_proposals(input_path: Path, output_path: Path, min_connector: float) -> None:
    payload = load_json(input_path)
    proposals = payload.get("proposals", [])
    filtered = [p for p in proposals if "__" in p.get("string", "") and p.get("connector", 0.0) >= min_connector]
    payload["proposals"] = filtered
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(output_path, payload)


def structural_twins(input_path: Path, output_path: Path, min_connector: float) -> None:
    payload = load_json(input_path)
    results = []
    for item in payload.get("results", []):
        token = item.get("string", "")
        connector = float(item.get("connector", 0.0))
        if "__" not in token or connector < min_connector:
            continue
        results.append(item)
    payload["results"] = results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(output_path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Structural-only filters for STM artefacts")
    sub = parser.add_subparsers(dest="command", required=True)

    strings_p = sub.add_parser("strings", help="Produce structural top strings list")
    strings_p.add_argument("state", type=Path)
    strings_p.add_argument("output", type=Path)
    strings_p.add_argument("--top", type=int, default=50)
    strings_p.add_argument("--min-occ", type=int, default=2)

    proposals_p = sub.add_parser("proposals", help="Filter proposal JSON to structural entries")
    proposals_p.add_argument("input", type=Path)
    proposals_p.add_argument("output", type=Path)
    proposals_p.add_argument("--min-connector", type=float, default=0.51)

    twins_p = sub.add_parser("twins", help="Filter twin JSON to structural entries")
    twins_p.add_argument("input", type=Path)
    twins_p.add_argument("output", type=Path)
    twins_p.add_argument("--min-connector", type=float, default=0.51)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "strings":
        structural_strings(args.state, args.output, top=args.top, min_occ=args.min_occ)
    elif args.command == "proposals":
        structural_proposals(args.input, args.output, min_connector=args.min_connector)
    elif args.command == "twins":
        structural_twins(args.input, args.output, min_connector=args.min_connector)


if __name__ == "__main__":
    main()
