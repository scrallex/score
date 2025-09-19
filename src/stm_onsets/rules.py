"""Mission-specific onset rule heuristics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class RuleHit:
    name: str
    score: float


def guardrail_rule(coh: float, ent: float, stab: float, thresholds: dict[str, float]) -> RuleHit | None:
    if (
        coh >= thresholds.get("min_coh", 0.0)
        and ent <= thresholds.get("max_ent", 1.0)
        and stab >= thresholds.get("min_stab", 0.0)
    ):
        return RuleHit("guardrail", 0.6)
    return None


def coherence_ramp(prev: float | None, current: float, delta: float = 5e-4) -> RuleHit | None:
    if prev is not None and current >= prev + delta:
        return RuleHit("coherence_ramp", 0.2)
    return None


def entropy_drop(prev: float | None, current: float, delta: float = 5e-4) -> RuleHit | None:
    if prev is not None and current <= prev - delta:
        return RuleHit("entropy_drop", 0.2)
    return None


def evaluate_mms_rules(prev_metrics: dict[str, float] | None, metrics: dict[str, float], thresholds: dict[str, float]) -> List[RuleHit]:
    coherence = metrics.get("coherence", 0.0)
    entropy = metrics.get("entropy", 1.0)
    stability = metrics.get("stability", 0.0)
    hits: List[RuleHit] = []
    guard_hit = guardrail_rule(coherence, entropy, stability, thresholds)
    if guard_hit:
        hits.append(guard_hit)
    if prev_metrics:
        for hit in (
            coherence_ramp(prev_metrics.get("coherence"), coherence),
            entropy_drop(prev_metrics.get("entropy"), entropy),
        ):
            if hit is not None:
                hits.append(hit)
    return hits


def score_hits(hits: Iterable[RuleHit]) -> float:
    return sum(hit.score for hit in hits)


__all__ = [
    "RuleHit",
    "evaluate_mms_rules",
    "score_hits",
]
