"""
FastAPI server scaffolding for the Sep Text Manifold.

This module defines a simple REST API exposing string metrics,
themes and raw event data.  It is intended as a starting point and
does not yet implement persistence or authentication.

To run the API locally install the ``fastapi`` and ``uvicorn``
packages and start the server with ``uvicorn sep_text_manifold.api:app``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class Metrics(BaseModel):
    coherence: float
    stability: float
    entropy: float
    rupture: float
    lambda_hazard: float


class StringMetricsOut(BaseModel):
    string: str
    metrics: Metrics


class ThemeOut(BaseModel):
    id: int
    members: List[str]


# In‑memory stores – these should be replaced by a proper database or
# state management layer in a production system.
_string_metrics: Dict[str, Metrics] = {}
_themes: List[List[str]] = []

app = FastAPI(title="Sep Text Manifold API")


@app.get("/strings/search", response_model=List[StringMetricsOut])
def search_strings(q: Optional[str] = None, min_patternability: float = 0.0) -> List[StringMetricsOut]:
    """Search for strings by substring and patternability threshold."""
    from .scoring import patternability_score
    results: List[StringMetricsOut] = []
    for s, metrics in _string_metrics.items():
        if q is None or q.lower() in s:
            p_score = patternability_score(metrics.coherence, metrics.stability, metrics.entropy, metrics.rupture)
            if p_score >= min_patternability:
                results.append(StringMetricsOut(string=s, metrics=metrics))
    return results


@app.get("/strings/{string}", response_model=StringMetricsOut)
def get_string(string: str) -> StringMetricsOut:
    """Get metrics for a single string."""
    metrics = _string_metrics.get(string)
    if metrics is None:
        raise HTTPException(status_code=404, detail="String not found")
    return StringMetricsOut(string=string, metrics=metrics)


@app.get("/themes", response_model=List[ThemeOut])
def list_themes() -> List[ThemeOut]:
    """List all detected themes."""
    return [ThemeOut(id=i, members=members) for i, members in enumerate(_themes)]


@app.get("/themes/{theme_id}", response_model=ThemeOut)
def get_theme(theme_id: int) -> ThemeOut:
    """Get a single theme by ID."""
    if 0 <= theme_id < len(_themes):
        return ThemeOut(id=theme_id, members=_themes[theme_id])
    raise HTTPException(status_code=404, detail="Theme not found")


@app.post("/load")
def load_data(strings: Dict[str, Dict[str, float]], themes: List[List[str]]) -> Dict[str, str]:
    """Load string metrics and themes into the API.

    This endpoint is a simple way to populate the in‑memory stores
    during development.  In a real application data would be loaded
    from a database or file system.
    """
    global _string_metrics, _themes
    _string_metrics = {s: Metrics(**m) for s, m in strings.items()}
    _themes = themes
    return {"status": "loaded", "strings": len(_string_metrics), "themes": len(_themes)}