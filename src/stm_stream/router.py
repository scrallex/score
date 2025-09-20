"""Streaming router skeleton for STM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class RouterConfig:
    coverage_min: float = 0.05
    coverage_max: float = 0.20
    max_path_dilution: float = 0.55
    max_signal_dilution: float = 0.60
    max_semantic_dilution: float = 0.70


class StreamingRouter:
    def __init__(self, config: RouterConfig) -> None:
        self.config = config

    def route(self, window: Dict[str, Any]) -> str:
        """Decide whether a window belongs in the foreground or deferred set."""
        dilution = window.get("dilution", {})
        path_value = float(dilution.get("path", 0.0))
        signal_value = float(dilution.get("signal", 0.0))
        semantic_value = float(dilution.get("semantic", dilution.get("semantic_dilution", 0.0)))

        if path_value > self.config.max_path_dilution:
            return "deferred"
        if signal_value > self.config.max_signal_dilution:
            return "deferred"
        if semantic_value and semantic_value > self.config.max_semantic_dilution:
            return "deferred"
        return "foreground"
