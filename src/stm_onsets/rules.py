"""Domain-specific onset rules with weak supervision support."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Iterable, List, Sequence

import numpy as np


@dataclass
class OnsetRule:
    name: str
    window: timedelta
    evaluator: Callable[[Sequence[float]], bool]


def bz_sign_change(window: Sequence[float]) -> bool:
    return bool(window and np.sign(window[0]) != np.sign(window[-1]))


def density_spike(window: Sequence[float], threshold: float = 0.2) -> bool:
    return bool(window and (max(window) - min(window) >= threshold))


MMS_RULES: List[OnsetRule] = [
    OnsetRule("bz_sign_change", timedelta(minutes=5), bz_sign_change),
]


def evaluate_rules(values: Sequence[float], rules: Iterable[OnsetRule]) -> List[str]:
    satisfied = []
    for rule in rules:
        if rule.evaluator(values):
            satisfied.append(rule.name)
    return satisfied
