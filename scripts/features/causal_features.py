"""Causal feature extraction helpers for STM guardrail experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


def _flatten_state(obj: Any, prefix: str = "") -> Dict[str, float]:
    """Convert nested mappings/sequences into a flat numerical view."""
    flat: Dict[str, float] = {}
    if obj is None:
        return flat
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten_state(value, path))
        return flat
    if isinstance(obj, (list, tuple, set)):
        for idx, value in enumerate(obj):
            path = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            flat.update(_flatten_state(value, path))
        return flat
    if isinstance(obj, bool):
        flat[prefix or "value"] = 1.0 if obj else 0.0
        return flat
    if isinstance(obj, (int, float)):
        flat[prefix or "value"] = float(obj)
        return flat
    if isinstance(obj, str):
        flat[prefix or "value"] = float(len(obj))
        return flat
    return flat


def _difference_score(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    total = 0.0
    for key in keys:
        total += abs(a.get(key, 0.0) - b.get(key, 0.0))
    return min(1.0, total / max(len(keys), 1))


def _velocity(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    deltas = [values[idx] - values[idx - 1] for idx in range(1, len(values))]
    magnitude = sum(abs(delta) for delta in deltas) / len(deltas)
    return min(1.0, abs(magnitude))


def _trend_acceleration(values: Sequence[float]) -> float:
    if len(values) < 3:
        return 0.0
    second_order = []
    for idx in range(2, len(values)):
        second_order.append(values[idx] - 2 * values[idx - 1] + values[idx - 2])
    magnitude = sum(abs(value) for value in second_order) / len(second_order)
    return min(1.0, abs(magnitude))


def _clamp(value: float) -> float:
    if math.isnan(value):  # pragma: no cover - defensive guard
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


@dataclass
class CausalFeatureExtractor:
    """Derive causal guardrail signals from STM windows."""

    coherence_floor: float = 1e-6

    def extract(
        self,
        window: Mapping[str, Any],
        *,
        history: Optional[Sequence[Mapping[str, Any]]] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, float]:
        metrics: Mapping[str, Any] = window.get("metrics", {}) if isinstance(window, Mapping) else {}
        dilution: Mapping[str, Any] = window.get("dilution", {}) if isinstance(window, Mapping) else {}
        coherence = float(metrics.get("coherence", 0.0))
        entropy = float(metrics.get("entropy", 0.0))
        stability = float(metrics.get("stability", 0.0))

        prev_window = history[-1] if history else None
        prev_metrics = prev_window.get("metrics", {}) if isinstance(prev_window, Mapping) else {}
        prev_dilution = prev_window.get("dilution", {}) if isinstance(prev_window, Mapping) else {}

        state_current = _flatten_state(window.get("state"))
        state_prev = _flatten_state(prev_window.get("state")) if isinstance(prev_window, Mapping) else {}
        context_state = _flatten_state((context or {}).get("state"))

        change_score = max(
            _difference_score(state_current, state_prev),
            _difference_score(state_current, context_state),
        )

        irreversible_actions = _clamp(change_score + (1.0 - stability) * 0.5)
        resource_commitment_ratio = _clamp(float(dilution.get("path", 0.0)))
        if resource_commitment_ratio <= 0.0 and context and "resources_locked" in context:
            resource_commitment_ratio = _clamp(float(context["resources_locked"]))

        decision_reversibility = _clamp(1.0 - resource_commitment_ratio)

        unsatisfied_preconditions = _clamp(entropy * (1.0 - coherence))
        effect_cascade_depth = _clamp(change_score + resource_commitment_ratio * 0.5)
        constraint_violation_distance = _clamp(1.0 - stability)

        coherence_history = [float(m.get("coherence", 0.0)) for m in _iter_metrics(history)]
        coherence_history.append(coherence)
        velocity = _velocity(coherence_history)
        divergence_rate = _trend_acceleration(coherence_history)

        entropy_history = [float(m.get("entropy", 0.0)) for m in _iter_metrics(history)]
        entropy_history.append(entropy)
        pattern_break = _velocity(entropy_history)

        return {
            "irreversible_actions": _clamp(irreversible_actions),
            "resource_commitment_ratio": _clamp(resource_commitment_ratio),
            "decision_reversibility": _clamp(decision_reversibility),
            "unsatisfied_preconditions": _clamp(unsatisfied_preconditions),
            "effect_cascade_depth": _clamp(effect_cascade_depth),
            "constraint_violation_distance": _clamp(constraint_violation_distance),
            "action_velocity": _clamp(velocity),
            "state_divergence_rate": _clamp(divergence_rate),
            "pattern_break_score": _clamp(pattern_break),
        }


def _iter_metrics(windows: Optional[Sequence[Mapping[str, Any]]]) -> Iterable[Mapping[str, Any]]:
    if not windows:
        return []
    return [window.get("metrics", {}) if isinstance(window, Mapping) else {} for window in windows]
