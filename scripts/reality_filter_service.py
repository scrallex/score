#!/usr/bin/env python3
"""FastAPI shim exposing the /seen interface over prepared truth-packs."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

from reality_filter import TruthPackEngine

app = FastAPI(
    title="Reality Filter /seen API",
    version="0.1.0",
    default_response_class=ORJSONResponse,
)


class SeenRequest(BaseModel):
    text: str = Field(..., description="Candidate span to validate")
    question: Optional[str] = Field(None, description="Optional question/context for the span")
    pack_manifest: str = Field(..., description="Path to manifest JSON produced by reality_filter_pack.py")
    seeds: Optional[List[str]] = Field(None, description="Override semantic seeds")
    semantic_threshold: float = Field(0.25, ge=0.0, le=1.0)
    structural_threshold: float = Field(0.46, ge=0.0, le=1.0)
    r_min: int = Field(2, ge=0)
    hazard_max: float = Field(0.55, ge=0.0, le=1.0)
    sigma_min: float = Field(0.28, ge=0.0, le=1.0)
    embedding_method: str = Field("hash")
    model: str = Field("all-MiniLM-L6-v2")
    hash_dims: int = Field(256, ge=1)
    embedding_min_occ: int = Field(1, ge=1)


class TwinPayload(BaseModel):
    string: str
    occurrences: int
    patternability: float
    semantic_similarity: float
    hazard: float


class SeenResponse(BaseModel):
    text: str
    question: Optional[str]
    pack: str
    decisions: Dict[str, bool]
    metrics: Dict[str, float]
    signature: Optional[str]
    twins: List[TwinPayload]
    action: str
    repair_candidate: Optional[str] = None


@lru_cache(maxsize=8)
def _load_engine(
    manifest_path: str,
    seeds_key: str,
    embedding_method: str,
    model: str,
    hash_dims: int,
    embedding_min_occ: int,
) -> TruthPackEngine:
    seeds = [s for s in seeds_key.split("|||" ) if s]
    seeds_to_use: List[str]
    if seeds:
        seeds_to_use = seeds
    else:
        manifest = Path(manifest_path)
        if not manifest.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest}")
        data = json.loads(manifest.read_text())
        seed_families = data.get("seed_families", {})
        seeds_to_use = data.get("seeds") or seed_families.get("factual", [])
    engine = TruthPackEngine.from_manifest(
        manifest_path,
        seeds=seeds_to_use,
        embedding_method=embedding_method,
        model_name=model,
        hash_dims=hash_dims,
        embedding_min_occ=embedding_min_occ,
        lru_size=200_000,
    )
    engine.prewarm(max_items=20_000)
    return engine


@app.post("/seen", response_model=SeenResponse)
def seen(request: SeenRequest) -> SeenResponse:
    manifest = Path(request.pack_manifest)
    if not manifest.exists():
        raise HTTPException(status_code=404, detail=f"Manifest not found: {manifest}")
    try:
        seeds_key = "|||".join(request.seeds or [])
        engine = _load_engine(
            str(manifest.resolve()),
            seeds_key,
            request.embedding_method,
            request.model,
            request.hash_dims,
            request.embedding_min_occ,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    eval_result = engine.evaluate_span(
        request.text,
        question=request.question,
        semantic_threshold=request.semantic_threshold,
        structural_threshold=request.structural_threshold,
        r_min=request.r_min,
        hazard_max=request.hazard_max,
        sigma_min=request.sigma_min,
    )

    action = "emit" if eval_result.admitted else "decline"
    repair_candidate = eval_result.repair_candidate.string if eval_result.repair_candidate else None

    return SeenResponse(
        text=eval_result.span,
        question=eval_result.question,
        pack=str(manifest.resolve()),
        decisions=eval_result.decisions(),
        metrics={**eval_result.metrics(), "repetitions": eval_result.occurrences},
        signature=eval_result.signature,
        twins=[
            TwinPayload(
                string=twin.string,
                occurrences=twin.occurrences,
                patternability=twin.patternability,
                semantic_similarity=twin.semantic_similarity,
                hazard=twin.hazard,
            )
            for twin in eval_result.twins
        ],
        action=action,
        repair_candidate=repair_candidate,
    )


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":  # pragma: no cover - manual run helper
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
