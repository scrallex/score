"""FastAPI service exposing canned STM demo payloads."""

from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

from datetime import datetime

import asyncio
import csv
import io
import math
import re
from itertools import cycle
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, Request
from starlette.responses import StreamingResponse

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, Spacer

from sep_text_manifold.manifold import build_manifold
from sep_text_manifold.strings import extract_strings

from demo.standalone import DEFAULT_OUTPUT, build_payload

app = FastAPI(title="STM Demo API", version="0.1.0")

_output_path = Path(os.environ.get("STM_DEMO_PAYLOAD", str(DEFAULT_OUTPUT))).resolve()
_payload_lock = threading.Lock()
_cached_payload: Dict[str, Any] = {}
_cached_mtime: float = 0.0
_live_signals: list[Dict[str, Any]] = []
_max_upload_bytes = int(os.environ.get("STM_ANALYZE_MAX_BYTES", 5 * 1024 * 1024))
_stream_interval = float(os.environ.get("STM_LIVE_INTERVAL_SECONDS", "0.25"))


def _write_payload_to_disk(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _generate_payload() -> Dict[str, Any]:
    payload = build_payload()
    try:
        _write_payload_to_disk(_output_path, payload)
    except OSError:
        pass
    return payload


def _load_payload(force: bool = False) -> Dict[str, Any]:
    global _cached_payload, _cached_mtime
    with _payload_lock:
        if force or not _cached_payload:
            _cached_payload = _generate_payload()
            _cached_mtime = time.time()
            return _cached_payload
        if _output_path.exists():
            mtime = _output_path.stat().st_mtime
            if mtime > _cached_mtime:
                try:
                    _cached_payload = json.loads(_output_path.read_text(encoding="utf-8"))
                    _cached_mtime = mtime
                except json.JSONDecodeError:
                    _cached_payload = _generate_payload()
                    _cached_mtime = time.time()
    return _cached_payload


def _load_live_signals() -> list[Dict[str, Any]]:
    global _live_signals
    if _live_signals:
        return _live_signals
    state_path = Path("analysis/mms_state.json")
    try:
        data = _load_json_file(state_path)
        signals = data.get("signals", [])
        prepared: list[Dict[str, Any]] = []
        for idx, signal in enumerate(signals):
            metrics = signal.get("metrics", {})
            prepared.append(
                {
                    "index": idx,
                    "signature": signal.get("signature"),
                    "coherence": float(metrics.get("coherence", 0.0)),
                    "stability": float(metrics.get("stability", 0.0)),
                    "entropy": float(metrics.get("entropy", 0.0)),
                    "lambda_hazard": float(metrics.get("lambda_hazard", metrics.get("rupture", 0.0))),
                }
            )
        if prepared:
            _live_signals = prepared
    except Exception:
            _live_signals = []
    return _live_signals


def _load_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _prepare_signal_payload(signal: Dict[str, Any], *, include_timestamp: bool = True) -> Dict[str, Any]:
    metrics = signal.get("metrics", {})
    payload = {
        "index": signal.get("index", signal.get("id")),
        "signature": signal.get("signature"),
        "metrics": {
            "coherence": float(metrics.get("coherence", signal.get("coherence", 0.0))),
            "stability": float(metrics.get("stability", signal.get("stability", 0.0))),
            "entropy": float(metrics.get("entropy", signal.get("entropy", 0.0))),
            "lambda_hazard": float(
                metrics.get("lambda_hazard", signal.get("lambda_hazard", metrics.get("rupture", 0.0)))
            ),
        },
    }
    if include_timestamp:
        payload["timestamp"] = time.time()
    return payload


@app.on_event("startup")
def _on_startup() -> None:
    _load_payload(force=True)


@app.get("/health")
def health() -> Dict[str, Any]:
    payload = _load_payload()
    return {"status": "ok", "generated_at": payload.get("generated_at")}


@app.get("/api/demo")
@app.get("/demo")
def get_demo_payload() -> Dict[str, Any]:
    return _load_payload()


@app.get("/api/demo/list")
@app.get("/demo/list")
def list_demos() -> Dict[str, Any]:
    payload = _load_payload()
    demos = payload.get("demos", {})
    return {"keys": sorted(demos.keys())}


@app.get("/api/demo/{demo_id}")
@app.get("/demo/{demo_id}")
def get_demo(demo_id: str) -> Dict[str, Any]:
    payload = _load_payload()
    demos = payload.get("demos", {})
    data = demos.get(demo_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"demo '{demo_id}' not found")
    return {"id": demo_id, "data": data}


@app.post("/api/demo/refresh")
@app.post("/demo/refresh")
def refresh_demo() -> Dict[str, Any]:
    payload = _load_payload(force=True)
    return {"status": "refreshed", "generated_at": payload.get("generated_at")}


@app.get("/api/meta")
@app.get("/meta")
def get_meta() -> Dict[str, Any]:
    payload = _load_payload()
    return {
        "generated_at": payload.get("generated_at"),
        "sources": payload.get("sources", {}),
        "assets": payload.get("assets", {}),
    }


@app.get("/api/assets/{key}")
@app.get("/assets/{key}")
def get_asset(key: str) -> Dict[str, Any]:
    payload = _load_payload()
    assets = payload.get("assets", {})
    path = assets.get(key)
    if not path:
        raise HTTPException(status_code=404, detail=f"asset '{key}' not found")
    return {"key": key, "path": path}


@app.get("/api/ping")
@app.get("/ping")
def ping() -> Dict[str, str]:
    return {"status": "alive"}


async def _live_stream() -> AsyncIterator[str]:
    signals = _load_live_signals()
    if not signals:
        # Fallback to payload metrics if state missing
        payload = _load_payload()
        pattern = payload.get("demos", {}).get("pattern_prophet", {})
        candidates = pattern.get("candidates", [])
        signals = [
            {
                "index": idx,
                "signature": cand.get("signature"),
                "coherence": float((cand.get("event_metrics") or {}).get("coherence", 0.0)),
                "stability": float((cand.get("event_metrics") or {}).get("stability", 0.0)),
                "entropy": float((cand.get("event_metrics") or {}).get("entropy", 0.0)),
                "lambda_hazard": float((cand.get("event_metrics") or {}).get("lambda_hazard", 0.0)),
            }
            for idx, cand in enumerate(candidates)
        ]
    if not signals:
        yield "data: {}\n\n"
        return
    for signal in cycle(signals):
        payload = _prepare_signal_payload(signal)
        yield f"data: {json.dumps(payload)}\n\n"
        await asyncio.sleep(max(_stream_interval, 0.05))


@app.get("/api/demo/live")
@app.get("/demo/live")
async def get_live_demo_stream() -> StreamingResponse:
    return StreamingResponse(_live_stream(), media_type="text/event-stream")


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    signals = _load_live_signals()
    if not signals:
        await websocket.close(code=1011, reason="No live signals available")
        return
    index = 0
    try:
        while True:
            signal = signals[index % len(signals)]
            payload = _prepare_signal_payload(signal, include_timestamp=False)
            await websocket.send_json(payload)
            index += 1
            await asyncio.sleep(max(_stream_interval, 0.05))
    except WebSocketDisconnect:
        return


@app.post("/api/analyze/quick")
@app.post("/analyze/quick")
async def quick_analyze(file: UploadFile = File(...)) -> Dict[str, Any]:
    if _max_upload_bytes <= 0:
        raise HTTPException(status_code=500, detail="Upload limits not configured")

    content = await file.read(_max_upload_bytes + 1)
    if len(content) > _max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded file exceeds {_max_upload_bytes} bytes limit",
        )

    decoded = content.decode("utf-8", errors="ignore")
    reader = csv.reader(io.StringIO(decoded))
    try:
        header = next(reader)
    except StopIteration:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    numeric_stats: Dict[int, Dict[str, float]] = {}
    samples: list[list[str]] = []
    row_count = 0
    max_samples = 5
    numeric_rows: list[Dict[int, float]] = []

    for row in reader:
        if not row:
            continue
        row_count += 1
        if len(samples) < max_samples:
            samples.append(row)
        row_numeric: Dict[int, float] = {}
        for idx, value in enumerate(row):
            stats = numeric_stats.setdefault(
                idx,
                {
                    "count": 0.0,
                    "mean": 0.0,
                    "m2": 0.0,
                    "is_numeric": True,
                },
            )
            if not stats["is_numeric"]:
                continue
            try:
                x = float(value)
            except (TypeError, ValueError):
                stats["is_numeric"] = False
                continue
            stats["count"] += 1
            delta = x - stats["mean"]
            stats["mean"] += delta / stats["count"]
            stats["m2"] += delta * (x - stats["mean"])
            row_numeric[idx] = x
        if row_numeric:
            numeric_rows.append(row_numeric)

    summary: list[Dict[str, Any]] = []
    for idx, stats in numeric_stats.items():
        if not stats["is_numeric"] or stats["count"] <= 1:
            continue
        variance = stats["m2"] / (stats["count"] - 1)
        summary.append(
            {
                "column": header[idx] if idx < len(header) else f"column_{idx}",
                "index": idx,
                "samples": int(stats["count"]),
                "mean": round(stats["mean"], 4),
                "std_dev": round(math.sqrt(max(variance, 0.0)), 4),
            }
        )

    if not summary:
        raise HTTPException(status_code=400, detail="Uploaded CSV did not contain stable numeric columns")

    recommendations = {
        "window_bytes": 1024 if row_count > 100 else 256,
        "stride": 512 if row_count > 100 else 128,
        "suggested_signatures": min(len(summary) * 2, 12),
    }

    # Build manifold using the actual quantum metrics pipeline
    lines: list[str] = []
    for row_numeric in numeric_rows:
        tokens: list[str] = []
        for entry in summary:
            idx = entry["index"]
            if idx not in row_numeric:
                continue
            value = row_numeric[idx]
            tokens.append(f"{entry['column']}:{value:.6f}")
        if tokens:
            lines.append(" ".join(tokens))

    if not lines:
        raise HTTPException(status_code=400, detail="Unable to construct manifold from the uploaded data")

    corpus_bytes = "\n".join(lines).encode("utf-8")
    window_bytes = int(recommendations["window_bytes"]) or 256
    stride = int(recommendations["stride"]) or max(window_bytes // 2, 1)
    try:
        signals = build_manifold(
            corpus_bytes,
            window_bytes=window_bytes,
            stride=stride,
            max_signals=512,
        )
    except Exception as exc:  # pragma: no cover - guard unexpected failures
        raise HTTPException(status_code=500, detail=f"Failed to compute manifold: {exc}") from exc

    signature_counter: Counter[str] = Counter()
    signature_metrics: Dict[str, Dict[str, float]] = {}
    for signal in signals:
        signature = signal.get("signature")
        if not signature:
            continue
        signature_counter[signature] += 1
        metrics = signal.get("metrics", {})
        agg = signature_metrics.setdefault(
            signature,
            {"coherence": 0.0, "stability": 0.0, "entropy": 0.0, "lambda_hazard": 0.0},
        )
        agg["coherence"] += float(metrics.get("coherence", 0.0))
        agg["stability"] += float(metrics.get("stability", 0.0))
        agg["entropy"] += float(metrics.get("entropy", 0.0))
        agg["lambda_hazard"] += float(metrics.get("lambda_hazard", signal.get("lambda_hazard", 0.0)))

    top_signatures: list[Dict[str, Any]] = []
    for signature, count in signature_counter.most_common(8):
        aggregates = signature_metrics[signature]
        top_signatures.append(
            {
                "signature": signature,
                "count": count,
                "mean_coherence": round(aggregates["coherence"] / count, 4),
                "mean_stability": round(aggregates["stability"] / count, 4),
                "mean_entropy": round(aggregates["entropy"] / count, 4),
                "mean_lambda": round(aggregates["lambda_hazard"] / count, 4),
            }
        )

    manifold_summary = {
        "total_windows": len(signals),
        "window_bytes": window_bytes,
        "stride": stride,
        "top_signatures": top_signatures,
        "sample_signals": [_prepare_signal_payload(sig, include_timestamp=False) for sig in signals[:10]],
    }

    return {
        "rows": row_count,
        "columns": header,
        "numeric_columns": [item["column"] for item in summary],
        "metrics": summary,
        "samples": samples,
        "recommendations": recommendations,
        "manifold": manifold_summary,
    }


@app.post("/api/analyze/text")
@app.post("/analyze/text")
async def analyze_text(request: Request) -> Dict[str, Any]:
    """Analyze pasted text using the quantum manifold pipeline."""
    try:
        payload = await request.json()
    except Exception as exc:  # pragma: no cover - guard malformed JSON
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {exc}") from exc

    text_content = str(payload.get("text", "")) if isinstance(payload, dict) else ""
    text_content = text_content.strip()
    if not text_content:
        raise HTTPException(status_code=400, detail="No text provided")

    max_text_length = int(os.environ.get("STM_TEXT_MAX_CHARS", 100_000))
    if len(text_content) > max_text_length:
        raise HTTPException(status_code=413, detail=f"Text exceeds {max_text_length} characters limit")

    occurrences = extract_strings(text_content, file_id="pasted_text")
    if not occurrences:
        raise HTTPException(status_code=400, detail="Unable to extract structural tokens from the provided text")

    token_stats: Dict[str, Dict[str, Any]] = {}
    for occ in occurrences:
        token = occ.string
        if len(token) < 2:
            continue
        stats = token_stats.setdefault(token, {"count": 0, "positions": []})
        stats["count"] += 1
        stats["positions"].append(int(occ.byte_start))

    data_bytes = text_content.encode("utf-8")
    if not data_bytes:
        raise HTTPException(status_code=400, detail="Text is empty after encoding")

    # Window sizing tuned for text content
    window_bytes = max(128, min(512, len(data_bytes) // 6 or 128))
    stride = max(32, window_bytes // 2)

    try:
        signals = build_manifold(
            data_bytes,
            window_bytes=window_bytes,
            stride=stride,
            max_signals=1024,
        )
    except Exception as exc:  # pragma: no cover - guard unexpected failures
        raise HTTPException(status_code=500, detail=f"Failed to compute manifold: {exc}") from exc

    pattern_scores: Dict[str, Dict[str, Any]] = {}
    for signal in signals:
        signature = signal.get("signature")
        if not signature:
            continue
        metrics = signal.get("metrics", {})
        coherence = float(metrics.get("coherence", signal.get("coherence", 0.0)))
        stability = float(metrics.get("stability", signal.get("stability", 0.0)))
        if coherence <= 0.01 or stability <= 0.45:
            continue
        start = int(signal.get("window_start", signal.get("index", 0) - window_bytes))
        end = int(signal.get("window_end", signal.get("index", 0)))
        start = max(0, start)
        end = min(len(text_content), max(start + 1, end))
        snippet = text_content[start:end].strip()
        if len(snippet) > 240:
            snippet = snippet[:240] + "…"
        entry = pattern_scores.setdefault(
            signature,
            {
                "signature": signature,
                "count": 0,
                "sum_coherence": 0.0,
                "sum_stability": 0.0,
                "positions": [],
                "snippets": [],
                "spans": [],
            },
        )
        entry["count"] += 1
        entry["sum_coherence"] += coherence
        entry["sum_stability"] += stability
        entry["positions"].append(start)
        if snippet and len(entry["snippets"]) < 3:
            entry["snippets"].append(snippet)
        if len(entry["spans"]) < 5:
            entry["spans"].append({"start": start, "end": end})

    top_patterns: List[Dict[str, Any]] = []
    for signature, stats in pattern_scores.items():
        count = stats["count"]
        top_patterns.append(
            {
                "signature": signature,
                "count": count,
                "avg_coherence": round(stats["sum_coherence"] / count, 4),
                "avg_stability": round(stats["sum_stability"] / count, 4),
                "positions": stats["positions"][:5],
                "spans": stats["spans"][:5],
                "sample_snippet": stats["snippets"][0] if stats["snippets"] else "",
            }
        )
    top_patterns.sort(key=lambda item: item["avg_coherence"] * item["count"], reverse=True)
    top_patterns = top_patterns[:10]

    # Phrase-level repeating sequences (2-5 word n-grams)
    phrase_stats: Dict[str, Dict[str, Any]] = {}
    word_pattern = re.compile(r"\b\w+\b")
    words: List[str] = []
    word_positions: List[int] = []
    for match in word_pattern.finditer(text_content):
        words.append(match.group())
        word_positions.append(match.start())

    max_ngram = min(5, len(words))
    for n in range(2, max_ngram + 1):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i : i + n]).lower()
            start_pos = word_positions[i]
            stats = phrase_stats.setdefault(phrase, {"count": 0, "positions": []})
            stats["count"] += 1
            stats["positions"].append(start_pos)

    repeating_sequences: List[Dict[str, Any]] = []
    for phrase, stats in phrase_stats.items():
        freq = stats["count"]
        if freq < 2:
            continue
        positions = sorted(stats["positions"])
        gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
        periodicity = None
        if gaps:
            avg_gap = sum(gaps) / len(gaps)
            variance = sum((gap - avg_gap) ** 2 for gap in gaps) / len(gaps)
            if avg_gap > 0 and variance < avg_gap:
                periodicity = avg_gap
        repeating_sequences.append(
            {
                "phrase": phrase,
                "frequency": freq,
                "periodicity": periodicity,
                "positions": positions[:5],
            }
        )

    repeating_sequences.sort(key=lambda item: item["frequency"], reverse=True)
    if not repeating_sequences:
        # Fallback to single tokens if no phrases repeat
        for token, stats in token_stats.items():
            if stats["count"] < 2:
                continue
            positions = sorted(stats["positions"])
            repeating_sequences.append(
                {
                    "phrase": token,
                    "frequency": stats["count"],
                    "periodicity": None,
                    "positions": positions[:5],
                }
            )
        repeating_sequences.sort(key=lambda item: item["frequency"], reverse=True)
    repeating_sequences = repeating_sequences[:10]

    structural_coverage = len(pattern_scores) / max(len(signals), 1)
    repeated_tokens = sum(stats["count"] for stats in token_stats.values() if stats["count"] > 1)
    repetition_ratio = repeated_tokens / max(len(occurrences), 1)
    text_metrics = {
        "total_characters": len(text_content),
        "total_tokens": len(occurrences),
        "unique_tokens": len(token_stats),
        "structural_coverage": round(structural_coverage, 4),
        "repetition_ratio": round(repetition_ratio, 4),
        "structural_signatures": len(pattern_scores),
    }

    interpretation = _interpret_text_patterns(text_metrics, top_patterns, repeating_sequences)

    return {
        "success": True,
        "metrics": text_metrics,
        "structural_patterns": top_patterns,
        "repeating_sequences": repeating_sequences,
        "manifold": {
            "total_windows": len(signals),
            "window_bytes": window_bytes,
            "stride": stride,
            "emit_threshold": 0.01,
        },
        "interpretation": interpretation,
    }


def _interpret_text_patterns(
    metrics: Dict[str, Any],
    patterns: List[Dict[str, Any]],
    repeating: List[Dict[str, Any]],
) -> List[str]:
    """Generate a human-readable interpretation for text analysis."""

    insights: List[str] = []
    coverage = metrics.get("structural_coverage", 0.0)
    repetition_ratio = metrics.get("repetition_ratio", 0.0)

    if coverage > 0.3:
        insights.append("High structural coherence – text shows strong organizational patterns.")
    elif coverage > 0.1:
        insights.append("Moderate structure detected – some repeatable patterns present.")
    else:
        insights.append("Low structural coherence – content appears loosely organized.")

    if repetition_ratio > 0.4:
        insights.append("Significant repetition detected – likely templated or formulaic content.")
    elif repetition_ratio > 0.2:
        insights.append("Notable repetition present across the text.")

    if patterns:
        lead = patterns[0]
        insights.append(
            f"Lead structural signature {lead['signature']} appears {lead['count']} times with coherence {lead['avg_coherence']:.3f}."
        )

    periodic_phrases = [item for item in repeating if item.get("periodicity")]
    if periodic_phrases:
        insights.append(
            f"Detected {len(periodic_phrases)} repeating phrases with regular spacing – indicates templated structure."
        )
    elif repeating:
        insights.append("Multiple repeating phrases detected across the text.")

    return insights
