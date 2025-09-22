#!/usr/bin/env python3
"""Augment STM state artefacts with derived feature sets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.features import CausalFeatureExtractor
from scripts.experiments.build_causal_domain import blend_metrics  # type: ignore


def load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"State file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def enrich_with_causal_features(state: Dict[str, Any]) -> int:
    extractor = CausalFeatureExtractor()
    signals = state.get("signals")
    if not isinstance(signals, list):
        raise ValueError("State does not contain a 'signals' list")

    history: List[Dict[str, Any]] = []
    updated = 0
    for window in signals:
        if not isinstance(window, dict):
            history.append({})
            continue
        features = extractor.extract(window, history=history)
        window.setdefault("features", {})["causal"] = features
        history.append(window)
        updated += 1
    return updated


def default_output_path(input_path: Path, suffix: str) -> Path:
    stem = input_path.stem
    return input_path.with_name(f"{stem}_{suffix}{input_path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich STM states with derived features")
    parser.add_argument("input", type=Path, help="State JSON produced by STM pipeline")
    parser.add_argument("--output", type=Path, help="Output path (defaults to <input>_causal.json)")
    parser.add_argument(
        "--features",
        choices=("causal",),
        default="causal",
        help="Feature set to compute",
    )
    parser.add_argument(
        "--blend-metrics",
        action="store_true",
        help="Blend causal features into coherence/entropy/stability metrics",
    )
    args = parser.parse_args()

    state = load_state(args.input)
    updated = 0
    if args.features == "causal":
        updated = enrich_with_causal_features(state)
        if args.blend_metrics:
            for window in state.get("signals", []):
                if not isinstance(window, dict):
                    continue
                features = window.get("features", {}).get("causal")
                metrics = window.get("metrics", {})
                if isinstance(features, dict) and isinstance(metrics, dict):
                    window["metrics"] = blend_metrics(metrics, features)

    output_path = args.output or default_output_path(args.input, args.features)
    write_state(output_path, state)

    print(json.dumps({"output": str(output_path), "windows": updated}, indent=2))


if __name__ == "__main__":
    main()
