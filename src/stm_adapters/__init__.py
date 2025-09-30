"""Adapter registry for STM ingestion."""

from __future__ import annotations

from typing import Dict

try:
    from .nasa_themis import ThemisAdapter
except Exception:  # pragma: no cover - optional dependency
    ThemisAdapter = None  # type: ignore[assignment]

from .pddl_trace import PDDLTraceAdapter

ADAPTERS: Dict[str, object] = {}

if ThemisAdapter is not None:  # pragma: no branch - conditional registry
    ADAPTERS["nasa_themis"] = ThemisAdapter()

ADAPTERS["pddl_trace"] = PDDLTraceAdapter()


def get_adapter(name: str):
    try:
        return ADAPTERS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown adapter '{name}'") from exc
