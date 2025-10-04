#!/usr/bin/env python3
"""FastAPI shim exposing the /seen interface over prepared truth-packs."""

from __future__ import annotations

import asyncio
import atexit
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import orjson
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import ORJSONResponse, Response

from reality_filter import TruthPackEngine
from reality_filter.engine import SpanEvaluation

app = FastAPI(
    title="Reality Filter /seen API",
    version="0.1.0",
    default_response_class=ORJSONResponse,
)


@dataclass(frozen=True)
class EngineConfig:
    embedding_method: str
    model: str
    hash_dims: int
    embedding_min_occ: int


@dataclass(frozen=True)
class EvalConfig:
    semantic_threshold: float
    structural_threshold: float
    r_min: int
    hazard_max: float
    sigma_min: float


@dataclass(frozen=True)
class SeenEvalRequest:
    text: str
    question: Optional[str]
    config: EvalConfig


@dataclass(frozen=True)
class ParsedSeenPayload:
    manifest: Path
    seeds: Tuple[str, ...]
    seeds_key: str
    engine_config: EngineConfig
    eval_request: SeenEvalRequest


@lru_cache(maxsize=8)
def _load_engine(manifest_path: str, seeds_key: str, engine_config: EngineConfig) -> TruthPackEngine:
    seeds = tuple(filter(None, seeds_key.split("|||")))
    seeds_to_use: List[str]
    if seeds:
        seeds_to_use = list(seeds)
    else:
        manifest = Path(manifest_path)
        if not manifest.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest}")
        data = orjson.loads(manifest.read_bytes())
        seeds_to_use = []
        if isinstance(data, dict):
            seeds_value = data.get("seeds")
            if isinstance(seeds_value, list):
                seeds_to_use = [s for s in seeds_value if isinstance(s, str)]
            if not seeds_to_use:
                seed_families = data.get("seed_families", {})
                if isinstance(seed_families, dict):
                    factual = seed_families.get("factual", [])
                    if isinstance(factual, list):
                        seeds_to_use = [s for s in factual if isinstance(s, str)]
    engine = TruthPackEngine.from_manifest(
        manifest_path,
        seeds=seeds_to_use,
        embedding_method=engine_config.embedding_method,
        model_name=engine_config.model,
        hash_dims=engine_config.hash_dims,
        embedding_min_occ=engine_config.embedding_min_occ,
        lru_size=200_000,
    )
    engine.prewarm(max_items=20_000)
    return engine


class SeenBatcher:
    def __init__(self, max_batch: int = 32, max_delay: float = 0.005) -> None:
        if os.environ.get("RF_DISABLE_BATCH") == "1":
            self._max_batch = 1
            self._max_delay = 0.0
        else:
            self._max_batch = max_batch
            self._max_delay = max_delay
        self._queue: "asyncio.Queue[Tuple[TruthPackEngine, SeenEvalRequest, asyncio.Future]]" = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None

    async def submit(self, engine: TruthPackEngine, payload: SeenEvalRequest) -> "SpanEvaluation":
        if self._worker_task is None or self._worker_task.done():
            loop = asyncio.get_running_loop()
            self._worker_task = loop.create_task(self._run())
        loop = asyncio.get_running_loop()
        future: "asyncio.Future" = loop.create_future()
        if self._max_batch == 1:
            config = payload.config
            evaluation = engine.evaluate_spans(
                [payload.text],
                questions=[payload.question],
                semantic_threshold=config.semantic_threshold,
                structural_threshold=config.structural_threshold,
                r_min=config.r_min,
                hazard_max=config.hazard_max,
                sigma_min=config.sigma_min,
                twins_needed=False,
            )[0]
            future.set_result(evaluation)
            return await future
        await self._queue.put((engine, payload, future))
        return await future

    async def _run(self) -> None:
        while True:
            engine, request, future = await self._queue.get()
            batch: List[Tuple[TruthPackEngine, SeenEvalRequest, asyncio.Future]] = [
                (engine, request, future)
            ]
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
        self, batch: Sequence[Tuple[TruthPackEngine, SeenEvalRequest, asyncio.Future]]
    ) -> None:
        grouped: Dict[
            Tuple[TruthPackEngine, EvalConfig],
            List[Tuple[TruthPackEngine, SeenEvalRequest, asyncio.Future]],
        ] = {}
        for item in batch:
            engine, payload, future = item
            key = (engine, payload.config)
            grouped.setdefault(key, []).append((engine, payload, future))

        loop = asyncio.get_running_loop()
        for key, items in grouped.items():
            engine, eval_config = key
            spans = [payload.text for _, payload, _ in items]
            questions = [payload.question for _, payload, _ in items]
            if not any(question is not None for question in questions):
                questions_payload: Optional[List[Optional[str]]] = None
            else:
                questions_payload = questions

            evaluate = partial(
                engine.evaluate_spans,
                spans,
                questions=questions_payload,
                semantic_threshold=eval_config.semantic_threshold,
                structural_threshold=eval_config.structural_threshold,
                r_min=eval_config.r_min,
                hazard_max=eval_config.hazard_max,
                sigma_min=eval_config.sigma_min,
                twins_needed=False,
            )
            try:
                evaluations = await loop.run_in_executor(EXECUTOR, evaluate)
            except Exception as exc:  # pragma: no cover - propagate evaluation failures
                for _, _, future in items:
                    if not future.done():
                        future.set_exception(exc)
                continue
            for evaluation, (_, _payload, future) in zip(evaluations, items):
                if not future.done():
                    future.set_result(evaluation)


EXECUTOR = ThreadPoolExecutor(max_workers=int(os.environ.get("RF_EXECUTOR_WORKERS", "8")))
_batcher = SeenBatcher()
COUNTERS_PATH = Path("results/seen_counters.json")
COUNTER_WRITE_INTERVAL = 100
_counter_last_written = 0
_last_counters: Dict[str, float] = {}


def _write_counters(data: Dict[str, float]) -> None:
    global _last_counters
    COUNTERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    COUNTERS_PATH.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))
    _last_counters.update(data)


@atexit.register
def _flush_counters_on_exit() -> None:
    if _last_counters:
        try:
            _write_counters(_last_counters)
        except Exception:
            pass


def _parse_payload(data: dict) -> ParsedSeenPayload:
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
    seeds: Tuple[str, ...]
    if isinstance(seeds_raw, list):
        seeds = tuple(s for s in seeds_raw if isinstance(s, str) and s)
    else:
        seeds = tuple()

    def _coerce_float(value: object, default: float, field: str) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - validation guard
            raise HTTPException(status_code=400, detail=f"Field '{field}' must be a number") from exc

    def _coerce_int(value: object, default: int, field: str) -> int:
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - validation guard
            raise HTTPException(status_code=400, detail=f"Field '{field}' must be an integer") from exc

    eval_config = EvalConfig(
        semantic_threshold=_coerce_float(data.get("semantic_threshold"), 0.25, "semantic_threshold"),
        structural_threshold=_coerce_float(data.get("structural_threshold"), 0.46, "structural_threshold"),
        r_min=_coerce_int(data.get("r_min"), 2, "r_min"),
        hazard_max=_coerce_float(data.get("hazard_max"), 0.55, "hazard_max"),
        sigma_min=_coerce_float(data.get("sigma_min"), 0.28, "sigma_min"),
    )

    embedding_method = data.get("embedding_method", "hash")
    model = data.get("model", "all-MiniLM-L6-v2")
    if not isinstance(embedding_method, str):
        raise HTTPException(status_code=400, detail="Field 'embedding_method' must be a string")
    if not isinstance(model, str):
        raise HTTPException(status_code=400, detail="Field 'model' must be a string")

    engine_config = EngineConfig(
        embedding_method=embedding_method,
        model=model,
        hash_dims=_coerce_int(data.get("hash_dims"), 256, "hash_dims"),
        embedding_min_occ=_coerce_int(data.get("embedding_min_occ"), 1, "embedding_min_occ"),
    )

    eval_request = SeenEvalRequest(text=text, question=question, config=eval_config)
    seeds_key = "|||".join(seeds)
    return ParsedSeenPayload(
        manifest=Path(manifest_path),
        seeds=seeds,
        seeds_key=seeds_key,
        engine_config=engine_config,
        eval_request=eval_request,
    )


@app.post("/seen")
async def seen(request: Request) -> Response:
    try:
        payload = await request.body()
        data = orjson.loads(payload)
    except Exception as exc:  # pragma: no cover - invalid JSON
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from exc

    parsed = _parse_payload(data)
    manifest = parsed.manifest
    if not manifest.exists():
        raise HTTPException(status_code=404, detail=f"Manifest not found: {manifest}")
    manifest_resolved = manifest.resolve()
    try:
        engine = _load_engine(
            str(manifest_resolved),
            parsed.seeds_key,
            parsed.engine_config,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    eval_result = await _batcher.submit(engine, parsed.eval_request)

    action = "emit" if eval_result.admitted else "decline"
    repair_candidate = eval_result.repair_candidate.string if eval_result.repair_candidate else None

    manifest_str = str(manifest_resolved)

    response_payload = {
        "text": eval_result.span,
        "question": eval_result.question,
        "pack": manifest_str,
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

    counters = engine.counters_snapshot()
    requests_total = int(counters.get("requests_total", 0))
    write_needed = False
    global _counter_last_written
    if requests_total and (
        _counter_last_written == 0 or requests_total - _counter_last_written >= COUNTER_WRITE_INTERVAL
    ):
        _counter_last_written = requests_total
        write_needed = True
    elif requests_total <= 10 and _counter_last_written == 0:
        _counter_last_written = requests_total
        write_needed = True
    if write_needed:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _write_counters, counters)

    return Response(content=orjson.dumps(response_payload), media_type="application/json")


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":  # pragma: no cover - manual run helper
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
