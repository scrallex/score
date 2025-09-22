"""Adapters that translate external telemetry into STM artefacts."""

from .real_world_adapter import RealWorldAdapter, NormalisedEvent

__all__ = ["RealWorldAdapter", "NormalisedEvent"]

