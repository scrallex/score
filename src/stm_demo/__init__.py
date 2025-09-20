"""Runtime helpers for serving the STM demo payload."""

from __future__ import annotations

from .api import app  # re-export for uvicorn discovery

__all__ = ["app"]
