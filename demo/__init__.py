"""Demo utilities and prepared payload generation for the Structural Intelligence Engine."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any


def get_demo_payload_path() -> Path:
    """Return the default location for the generated demo payload JSON."""
    return Path(__file__).resolve().parent / "demo_payload.json"


__all__ = ["get_demo_payload_path"]
