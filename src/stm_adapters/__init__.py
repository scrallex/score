"""Adapter registry for STM ingestion."""

from __future__ import annotations

from typing import Dict

from .nasa_themis import ThemisAdapter

ADAPTERS: Dict[str, object] = {
    "nasa_themis": ThemisAdapter(),
}


def get_adapter(name: str):
    try:
        return ADAPTERS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown adapter '{name}'") from exc
