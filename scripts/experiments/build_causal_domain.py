#!/usr/bin/env python3
"""Generate a causal-feature-enriched STM domain for experimentation."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Mapping, List

sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.features import (
    CausalFeatureExtractor,
    build_logistics_features,
    native_metrics_provider as logistics_native_provider,
)
from sep_text_manifold import native
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

    logistics_irrev = float(features.get("logistics_irreversibility", 0.0))
    logistics_momentum = float(features.get("logistics_momentum", 0.5))
    logistics_entropy = float(features.get("logistics_cluster_entropy", 0.0))
    logistics_balance = float(features.get("logistics_predicate_balance", 0.5))
    logistics_delta = float(features.get("logistics_predicate_delta", 0.0))

    momentum_delta = logistics_momentum - 0.5
    balance_delta = logistics_balance - 0.5

    adjusted_coherence = _clamp(
        coherence
        + 7e-4 * resource
        + 4e-4 * irreversible
        + 1.8e-3 * logistics_irrev
        + 1.0e-3 * momentum_delta
        + 7e-4 * balance_delta
        - 8e-4 * logistics_delta
    )
    adjusted_entropy = _clamp(
        entropy
        - 0.05 * irreversible
        + 0.04 * (divergence + features.get("pattern_break_score", 0.0))
        - 0.07 * logistics_irrev
        + 0.05 * (0.5 - momentum_delta)
        + 0.03 * logistics_entropy
        + 0.015 * logistics_delta
    )
    adjusted_stability = _clamp(
        stability
        + 0.05 * (1.0 - constraint)
        - 0.04 * divergence
        + 0.05 * logistics_momentum
        - 0.03 * logistics_entropy
        + 0.01 * balance_delta
    )

    return {
        **metrics,
        "coherence": adjusted_coherence,
        "entropy": adjusted_entropy,
        "stability": adjusted_stability,
    }


def enrich_state_file(
    path: Path,
    extractor: CausalFeatureExtractor,
    include_logistics: bool,
    *,
    metrics_provider=None,
) -> None:
    data = load_json(path)
    signals = data.get("signals")
    if not isinstance(signals, list):
        return
    logistics_payload: list[Mapping[str, float]] = []
    if include_logistics:
        try:
            logistics_payload = build_logistics_features(data, metrics_provider=metrics_provider)
        except Exception:
            logistics_payload = []
    history = []
    for window in signals:
        if not isinstance(window, dict):
            history.append({})
            continue
        features = extractor.extract(window, history=history)
        buckets = window.setdefault("features", {})
        buckets["causal"] = features
        if include_logistics and logistics_payload:
            idx = len(history)
            if idx < len(logistics_payload):
                buckets["logistics"] = logistics_payload[idx]
                merged_features = {**features, **logistics_payload[idx]}
            else:
                merged_features = dict(features)
        else:
            merged_features = dict(features)
        metrics = window.get("metrics", {})
        window["metrics"] = blend_metrics(metrics, merged_features)
        history.append(window)
    dump_json(path, data)


def enrich_domain_states(destination: Path, include_logistics: bool, *, metrics_provider=None) -> None:
    extractor = CausalFeatureExtractor()
    for subset in ("invalid", "gold"):
        states_dir = destination / subset / "states"
        if not states_dir.exists():
            continue
        for state_path in sorted(states_dir.glob("*.json")):
            enrich_state_file(state_path, extractor, include_logistics, metrics_provider=metrics_provider)


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct causal-feature enriched domain")
    parser.add_argument("source", type=Path, help="Existing STM domain root (e.g. output/planbench_by_domain/logistics)")
    parser.add_argument("destination", type=Path, help="Destination directory for causal-enriched domain")
    parser.add_argument(
        "--aggregated-state",
        type=Path,
        help="Optional aggregated state JSON to copy into the destination",
    )
    parser.add_argument(
        "--include-logistics",
        action="store_true",
        help="Also compute logistics-specific features and blend them into metrics",
    )
    parser.add_argument(
        "--use-native-quantum",
        action="store_true",
        help="Prefer the native QFH/QBSA engine when available",
    )
    args = parser.parse_args()

    native.set_use_native(args.use_native_quantum)

    copy_domain(args.source, args.destination)

    aggregated_path = None
    if args.aggregated_state and args.aggregated_state.exists():
        target_path = args.destination / args.aggregated_state.name
        shutil.copy2(args.aggregated_state, target_path)
        aggregated_path = target_path

    metrics_provider = logistics_native_provider if args.use_native_quantum else None

    enrich_domain_states(
        args.destination,
        include_logistics=args.include_logistics,
        metrics_provider=metrics_provider,
    )

    if aggregated_path and aggregated_path.exists():
        extractor = CausalFeatureExtractor()
        enrich_state_file(
            aggregated_path,
            extractor,
            include_logistics=args.include_logistics,
            metrics_provider=metrics_provider,
        )


if __name__ == "__main__":
    main()
