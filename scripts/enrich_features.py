#!/usr/bin/env python3
"""Augment STM state artefacts with derived feature sets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.features import CausalFeatureExtractor, build_logistics_features
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


def enrich_with_logistics_features(state: Dict[str, Any]) -> int:
    signals = state.get("signals")
    if not isinstance(signals, list):
        raise ValueError("State does not contain a 'signals' list")
    settings = state.get("settings", {})
    token_dir_value = settings.get("directory")
    if not token_dir_value:
        raise ValueError("Settings directory missing; cannot locate token streams")
    token_dir = Path(token_dir_value)
    if not token_dir.exists():
        raise FileNotFoundError(f"Token directory not found: {token_dir}")

    features = build_logistics_features(state)

    if len(features) != len(signals):
        raise ValueError(
            f"Token window count ({len(features)}) does not match signal count ({len(signals)})"
        )

    for window, feats in zip(signals, features):
        logistics_bucket = window.setdefault("features", {}).setdefault("logistics", {})
        logistics_bucket.update(feats)
    return len(features)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich STM states with derived features")
    parser.add_argument("input", type=Path, help="State JSON produced by STM pipeline")
    parser.add_argument("--output", type=Path, help="Output path (defaults to <input>_causal.json)")
    parser.add_argument(
        "--features",
        choices=("causal", "logistics"),
        nargs="+",
        default=["causal"],
        help="Feature sets to compute",
    )
    parser.add_argument(
        "--blend-metrics",
        action="store_true",
        help="Blend causal features into coherence/entropy/stability metrics",
    )
    args = parser.parse_args()

    state = load_state(args.input)
    total_windows = 0

    if "causal" in args.features:
        enrich_with_causal_features(state)
    if "logistics" in args.features:
        enrich_with_logistics_features(state)

    if args.blend_metrics:
        for window in state.get("signals", []):
            if not isinstance(window, dict):
                continue
            metrics = window.get("metrics", {})
            if not isinstance(metrics, dict):
                continue
            causal_feats = window.get("features", {}).get("causal", {})
            logistics_feats = window.get("features", {}).get("logistics", {})
            combined: Dict[str, float] = {}
            if isinstance(causal_feats, dict):
                combined.update(causal_feats)
            if isinstance(logistics_feats, dict):
                combined.update(logistics_feats)
            if combined:
                window["metrics"] = blend_metrics(metrics, combined)

    output_path = args.output or default_output_path(args.input, args.features)
    write_state(output_path, state)

    total_windows = len(state.get("signals", []))
    print(json.dumps({"output": str(output_path), "windows": total_windows}, indent=2))


if __name__ == "__main__":
    main()
