"""Streaming router skeleton for STM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List


@dataclass
class RouterConfig:
    coverage_min: float = 0.05
    coverage_max: float = 0.20


class StreamingRouter:
    def __init__(self, config: RouterConfig) -> None:
        self.config = config

    def route(self, window: Dict[str, Any]) -> str:
        """Placeholder routing logic."""
        return "foreground"
