#!/usr/bin/env python3
"""Generate a causal-feature-enriched STM domain for experimentation."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Mapping

sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.features import CausalFeatureExtractor
from scripts.features.causal_features import _clamp


def copy_domain(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Mapping) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def blend_metrics(metrics: Mapping[str, float], features: Mapping[str, float]) -> Dict[str, float]:
    coherence = float(metrics.get("coherence", 0.0))
    entropy = float(metrics.get("entropy", 1.0))
    stability = float(metrics.get("stability", 0.0))

    resource = float(features.get("resource_commitment_ratio", 0.0))
    irreversible = float(features.get("irreversible_actions", 0.0))
    divergence = float(features.get("state_divergence_rate", 0.0))
    constraint = float(features.get("constraint_violation_distance", 0.0))

    adjusted_coherence = _clamp(coherence + 5e-4 * resource + 3e-4 * irreversible)
    adjusted_entropy = _clamp(entropy - 0.05 * irreversible + 0.04 * (divergence + features.get("pattern_break_score", 0.0)))
    adjusted_stability = _clamp(stability + 0.05 * (1.0 - constraint) - 0.04 * divergence)

    return {
        **metrics,
        "coherence": adjusted_coherence,
        "entropy": adjusted_entropy,
        "stability": adjusted_stability,
    }


def enrich_state_file(path: Path, extractor: CausalFeatureExtractor) -> None:
    data = load_json(path)
    signals = data.get("signals")
    if not isinstance(signals, list):
        return
    history = []
    for window in signals:
        if not isinstance(window, dict):
            history.append({})
            continue
        features = extractor.extract(window, history=history)
        window.setdefault("features", {})["causal"] = features
        metrics = window.get("metrics", {})
        window["metrics"] = blend_metrics(metrics, features)
        history.append(window)
    dump_json(path, data)


def enrich_domain_states(destination: Path) -> None:
    extractor = CausalFeatureExtractor()
    for subset in ("invalid", "gold"):
        states_dir = destination / subset / "states"
        if not states_dir.exists():
            continue
        for state_path in sorted(states_dir.glob("*.json")):
            enrich_state_file(state_path, extractor)


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct causal-feature enriched domain")
    parser.add_argument("source", type=Path, help="Existing STM domain root (e.g. output/planbench_by_domain/logistics)")
    parser.add_argument("destination", type=Path, help="Destination directory for causal-enriched domain")
    parser.add_argument(
        "--aggregated-state",
        type=Path,
        help="Optional aggregated state JSON to copy into the destination",
    )
    args = parser.parse_args()

    copy_domain(args.source, args.destination)

    if args.aggregated_state and args.aggregated_state.exists():
        target_path = args.destination / args.aggregated_state.name
        shutil.copy2(args.aggregated_state, target_path)

    enrich_domain_states(args.destination)


if __name__ == "__main__":
    main()
