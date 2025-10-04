#!/usr/bin/env python3
"""FastAPI shim exposing the /seen interface over prepared truth-packs."""

from __future__ import annotations

import asyncio
import json
import os
import time
from functools import lru_cache, partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import orjson
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import ORJSONResponse

from reality_filter import TruthPackEngine

app = FastAPI(
    title="Reality Filter /seen API",
    version="0.1.0",
    default_response_class=ORJSONResponse,
)


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


class SeenBatcher:
    def __init__(self, max_batch: int = 128, max_delay: float = 0.010) -> None:
        if os.environ.get("RF_DISABLE_BATCH") == "1":
            self._max_batch = 1
            self._max_delay = 0.0
        else:
            self._max_batch = max_batch
            self._max_delay = max_delay
        self._queue: "asyncio.Queue[Tuple[TruthPackEngine, Dict[str, object], asyncio.Future]]" = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None

    async def submit(self, engine: TruthPackEngine, payload: Dict[str, object]) -> "SpanEvaluation":
        if self._worker_task is None or self._worker_task.done():
            loop = asyncio.get_running_loop()
            self._worker_task = loop.create_task(self._run())
        loop = asyncio.get_running_loop()
        future: "asyncio.Future" = loop.create_future()
        if self._max_batch == 1:
            evaluation = engine.evaluate_spans(
                [payload["text"]],
                questions=[payload.get("question")],
                semantic_threshold=float(payload["semantic_threshold"]),
                structural_threshold=float(payload["structural_threshold"]),
                r_min=int(payload["r_min"]),
                hazard_max=float(payload["hazard_max"]),
                sigma_min=float(payload["sigma_min"]),
                twins_needed=[False],
            )[0]
            future.set_result(evaluation)
            return await future
        await self._queue.put((engine, payload, future))
        return await future

    async def _run(self) -> None:
        while True:
            engine, request, future = await self._queue.get()
            batch: List[Tuple[TruthPackEngine, SeenRequest, asyncio.Future]] = [(engine, request, future)]
            start = time.perf_counter()
            while len(batch) < self._max_batch:
                remaining = self._max_delay - (time.perf_counter() - start)
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                batch.append(item)
            await self._process_batch(batch)

    async def _process_batch(
        self, batch: Sequence[Tuple[TruthPackEngine, SeenRequest, asyncio.Future]]
    ) -> None:
        grouped: Dict[
            Tuple[TruthPackEngine, float, float, int, float, float],
            List[Tuple[TruthPackEngine, Dict[str, object], asyncio.Future]],
        ] = {}
        for item in batch:
            engine, payload, future = item
            key = (
                engine,
                float(payload["semantic_threshold"]),
                float(payload["structural_threshold"]),
                int(payload["r_min"]),
                float(payload["hazard_max"]),
                float(payload["sigma_min"]),
            )
            grouped.setdefault(key, []).append((engine, payload, future))

        loop = asyncio.get_running_loop()
        for key, items in grouped.items():
            engine = key[0]
            spans = [payload["text"] for _, payload, _ in items]
            questions_raw = [payload.get("question") for _, payload, _ in items]
            questions: Optional[List[Optional[str]]] = (
                questions_raw if any(q is not None for q in questions_raw) else None
            )
            twins_flags = [False] * len(items)

            evaluate = partial(
                engine.evaluate_spans,
                spans,
                questions=questions,
                semantic_threshold=key[1],
                structural_threshold=key[2],
                r_min=key[3],
                hazard_max=key[4],
                sigma_min=key[5],
                twins_needed=twins_flags,
            )
            try:
                evaluations = await loop.run_in_executor(None, evaluate)
            except Exception as exc:  # pragma: no cover - propagate evaluation failures
                for _, _, future in items:
                    if not future.done():
                        future.set_exception(exc)
                continue
            for evaluation, (_, _payload, future) in zip(evaluations, items):
                if not future.done():
                    future.set_result(evaluation)


_batcher = SeenBatcher()


def _parse_payload(data: dict) -> Tuple[str, Optional[str], Path, List[str], Dict[str, object]]:
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Payload must be a JSON object")
    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=400, detail="Field 'text' is required")
    question = data.get("question")
    if question is not None and not isinstance(question, str):
        raise HTTPException(status_code=400, detail="Field 'question' must be a string")
    manifest_path = data.get("pack_manifest")
    if not isinstance(manifest_path, str) or not manifest_path:
        raise HTTPException(status_code=400, detail="Field 'pack_manifest' is required")
    seeds_raw = data.get("seeds")
    seeds = [s for s in seeds_raw if isinstance(s, str)] if isinstance(seeds_raw, list) else []
    thresholds = {
        "semantic_threshold": float(data.get("semantic_threshold", 0.25)),
        "structural_threshold": float(data.get("structural_threshold", 0.46)),
        "r_min": int(data.get("r_min", 2)),
        "hazard_max": float(data.get("hazard_max", 0.55)),
        "sigma_min": float(data.get("sigma_min", 0.28)),
    }
    embed_cfg = {
        "embedding_method": data.get("embedding_method", "hash"),
        "model": data.get("model", "all-MiniLM-L6-v2"),
        "hash_dims": int(data.get("hash_dims", 256)),
        "embedding_min_occ": int(data.get("embedding_min_occ", 1)),
    }
    return text, question, Path(manifest_path), seeds, {**thresholds, **embed_cfg}


@app.post("/seen")
async def seen(request: Request) -> ORJSONResponse:
    try:
        payload = await request.body()
        data = orjson.loads(payload)
    except Exception as exc:  # pragma: no cover - invalid JSON
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from exc

    text, question, manifest, seeds, config = _parse_payload(data)
    if not manifest.exists():
        raise HTTPException(status_code=404, detail=f"Manifest not found: {manifest}")
    try:
        seeds_key = "|||".join(seeds)
        engine = _load_engine(
            str(manifest.resolve()),
            seeds_key,
            config["embedding_method"],
            config["model"],
            config["hash_dims"],
            config["embedding_min_occ"],
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    request_payload = {
        "text": text,
        "question": question,
        "semantic_threshold": config["semantic_threshold"],
        "structural_threshold": config["structural_threshold"],
        "r_min": config["r_min"],
        "hazard_max": config["hazard_max"],
        "sigma_min": config["sigma_min"],
    }

    eval_result = await _batcher.submit(engine, request_payload)

    action = "emit" if eval_result.admitted else "decline"
    repair_candidate = eval_result.repair_candidate.string if eval_result.repair_candidate else None

    response_payload = {
        "text": eval_result.span,
        "question": eval_result.question,
        "pack": str(manifest.resolve()),
        "decisions": eval_result.decisions(),
        "metrics": {**eval_result.metrics(), "repetitions": eval_result.occurrences},
        "signature": eval_result.signature,
        "twins": [
            {
                "string": twin.string,
                "occurrences": twin.occurrences,
                "patternability": twin.patternability,
                "semantic_similarity": twin.semantic_similarity,
                "hazard": twin.hazard,
                "source": twin.source,
            }
            for twin in eval_result.twins
        ],
        "action": action,
        "repair_candidate": repair_candidate,
    }

    return ORJSONResponse(response_payload)


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":  # pragma: no cover - manual run helper
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
