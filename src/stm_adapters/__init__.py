"""Adapter registry for STM ingestion."""

from __future__ import annotations

from typing import Dict

from .nasa_themis import ThemisAdapter
from .pddl_trace import PDDLTraceAdapter

ADAPTERS: Dict[str, object] = {
    "nasa_themis": ThemisAdapter(),
    "pddl_trace": PDDLTraceAdapter(),
}


def get_adapter(name: str):
    try:
        return ADAPTERS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown adapter '{name}'") from exc
