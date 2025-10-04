"""Simplified guardrail utilities for the backtester test harness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class PathMetrics:
    """Container capturing structural metrics used for throttling."""

    entropy: float
    coherence: float
    stability: float
    rupture: float
    hazard: float

    def clamp(self) -> "PathMetrics":
        return PathMetrics(
            entropy=max(0.0, float(self.entropy)),
            coherence=max(0.0, float(self.coherence)),
            stability=max(0.0, float(self.stability)),
            rupture=max(0.0, float(self.rupture)),
            hazard=max(0.0, float(self.hazard)),
        )


def throttle_factor(current: PathMetrics, previous: Optional[PathMetrics] = None) -> float:
    """Compute a position size between 0 and 1.

    The heuristic favours coherent, stable paths with low entropy/hazard
    and reduces size when structural metrics deteriorate from the
    previous observation.  The exact numbers are not sourced from the
    production system; they just need to be deterministic for tests.
    """

    metrics = current.clamp()
    base = (metrics.coherence + metrics.stability) / 2.0
    base = max(0.0, min(base, 1.5))

    entropy_penalty = 1.0 - min(metrics.entropy / 5.0, 1.0)
    hazard_penalty = 1.0 - min(metrics.hazard, 1.0)
    score = base * entropy_penalty * hazard_penalty

    if previous is not None:
        prev = previous.clamp()
        drift = abs(metrics.coherence - prev.coherence) + abs(metrics.stability - prev.stability)
        drift_penalty = 1.0 / (1.0 + drift)
        score *= drift_penalty

    return max(0.0, min(score, 1.0))


__all__ = ["PathMetrics", "throttle_factor"]
