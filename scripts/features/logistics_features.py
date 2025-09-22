"""Logistics-specific feature construction based on string scores."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


IRREVERSIBLE_WEIGHTS: Dict[str, float] = {
    "deliver": 1.25,
    "unload": 1.1,
    "load": 0.65,
    "drive": 0.4,
    "fly": 0.45,
}

CLUSTER_MAP: Dict[str, str] = {
    "load": "loading_ops",
    "drive": "movement_ops",
    "unload": "delivery_ops",
    "deliver": "delivery_ops",
    "fly": "movement_ops",
}

POSITIVE_PREFIXES: Iterable[str] = ("at_", "in_", "vehicle_", "available", "ready")
NEGATIVE_PREFIXES: Iterable[str] = ("not_at_", "not_in_", "not_vehicle_", "missing", "blocked")


def _initialise_cluster_store(size: int) -> Dict[str, List[float]]:
    clusters = set(CLUSTER_MAP.values()) | {"other_ops"}
    return {cluster: [0.0] * size for cluster in clusters}


def _entropy(values: Iterable[float]) -> float:
    total = sum(values)
    if total <= 0.0:
        return 0.0
    probs = [v / total for v in values if v > 0.0]
    if not probs:
        return 0.0
    return -sum(p * math.log(p, 2) for p in probs)


def build_logistics_features(state: Mapping[str, Any]) -> List[Dict[str, float]]:
    signals = state.get("signals") or []
    total_windows = len(signals)
    if total_windows == 0:
        return []

    irreversible_scores = [0.0] * total_windows
    action_counts = [0.0] * total_windows
    positive_counts = [0.0] * total_windows
    negative_counts = [0.0] * total_windows
    cluster_store = _initialise_cluster_store(total_windows)

    string_scores: Mapping[str, Mapping[str, Any]] = state.get("string_scores", {})  # type: ignore[assignment]
    for token, payload in string_scores.items():
        window_ids = payload.get("window_ids") or []
        if not window_ids:
            continue
        weight = IRREVERSIBLE_WEIGHTS.get(token)
        if weight is not None:
            cluster = CLUSTER_MAP.get(token, "other_ops")
            cluster_array = cluster_store.setdefault(cluster, [0.0] * total_windows)
            for idx in window_ids:
                if 0 <= idx < total_windows:
                    irreversible_scores[idx] += weight
                    action_counts[idx] += 1.0
                    cluster_array[idx] += 1.0

        if any(token.startswith(prefix) for prefix in POSITIVE_PREFIXES):
            for idx in window_ids:
                if 0 <= idx < total_windows:
                    positive_counts[idx] += 1.0
        if any(token.startswith(prefix) for prefix in NEGATIVE_PREFIXES):
            for idx in window_ids:
                if 0 <= idx < total_windows:
                    negative_counts[idx] += 1.0

    # Derive per-window features
    irreversibility_series: List[float] = []
    momentum_series: List[float] = []
    cluster_entropy_series: List[float] = []
    predicate_balance_series: List[float] = []
    predicate_delta_series: List[float] = []

    max_weight = max(IRREVERSIBLE_WEIGHTS.values()) or 1.0
    deltas: List[float] = []
    for idx in range(total_windows):
        if action_counts[idx] > 0:
            value = irreversible_scores[idx] / (action_counts[idx] * max_weight)
        else:
            value = 0.0
        irreversibility_series.append(clamp(value))

        pos = positive_counts[idx]
        neg = negative_counts[idx]
        total_predicates = pos + neg
        if total_predicates > 0:
            signed = (pos - neg) / total_predicates
        else:
            signed = 0.0
        delta = (-signed)  # negative pressure indicates increasing blockers
        deltas.append(delta)
        predicate_balance_series.append(clamp(0.5 + 0.5 * signed))
        predicate_delta_series.append(abs(delta))

        # Cluster entropy uses counts across clusters plus inferred "other" bucket.
        cluster_values = [bucket[idx] for bucket in cluster_store.values()]
        other = max(action_counts[idx] - sum(cluster_values), 0.0)
        cluster_values_with_other = cluster_values[:-1] + [cluster_values[-1] + other]
        ent = _entropy(cluster_values_with_other)
        max_ent = math.log(max(len([v for v in cluster_values_with_other if v > 0]), 1), 2)
        cluster_entropy_series.append(clamp(ent / max(max_ent, 1.0)))

    for idx, delta in enumerate(deltas):
        if idx == 0:
            momentum = clamp(0.5 + 0.6 * delta)
        elif idx == 1:
            diff = delta - deltas[idx - 1]
            momentum = clamp(0.5 + 0.55 * diff)
        else:
            accel = (delta - deltas[idx - 1]) - (deltas[idx - 1] - deltas[idx - 2])
            momentum = clamp(0.5 + 0.4 * accel)
        momentum_series.append(momentum)

    features: List[Dict[str, float]] = []
    for irr, mom, ent, balance, delta in zip(
        irreversibility_series,
        momentum_series,
        cluster_entropy_series,
        predicate_balance_series,
        predicate_delta_series,
    ):
        features.append(
            {
                "logistics_irreversibility": irr,
                "logistics_momentum": mom,
                "logistics_cluster_entropy": ent,
                "logistics_predicate_balance": balance,
                "logistics_predicate_delta": clamp(delta),
            }
        )
    return features
