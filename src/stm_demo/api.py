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

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, Spacer, TableStyle
from xml.sax.saxutils import escape

from sep_text_manifold.manifold import build_manifold
from sep_text_manifold.strings import extract_strings

from demo.standalone import DEFAULT_OUTPUT, build_payload

app = FastAPI(title="STM Demo API", version="0.1.0")

_styles = getSampleStyleSheet()
TITLE_STYLE = ParagraphStyle(
    "STMTitle",
    parent=_styles["Title"],
    fontSize=18,
    leading=22,
    alignment=1,
    spaceAfter=12,
)
DATE_STYLE = ParagraphStyle(
    "STMDate",
    parent=_styles["Normal"],
    fontSize=9,
    leading=12,
    textColor=colors.HexColor("#7d8193"),
    spaceAfter=12,
)
HEADING_STYLE = ParagraphStyle(
    "STMHeading",
    parent=_styles["Heading2"],
    fontSize=12,
    leading=16,
    spaceBefore=14,
    spaceAfter=6,
)
BODY_STYLE = ParagraphStyle(
    "STMBody",
    parent=_styles["BodyText"],
    fontSize=10,
    leading=14,
    spaceAfter=6,
)
QUOTE_STYLE = ParagraphStyle(
    "STMQuote",
    parent=_styles["BodyText"],
    fontSize=9,
    leading=13,
    textColor=colors.HexColor("#a7aac1"),
    leftIndent=12,
    spaceAfter=6,
    fontName="Helvetica-Oblique",
)

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
                "contexts": [],
                "context_groups": {},
                "spans": [],
                "bounds": [],
            },
        )
        entry["count"] += 1
        entry["sum_coherence"] += coherence
        entry["sum_stability"] += stability
        entry["positions"].append(start)
        if snippet and len(entry["snippets"]) < 3:
            entry["snippets"].append(snippet)
        if len(entry["spans"]) < 50:
            entry["spans"].append({"start": start, "end": end})
        context_window = 120
        context_slice = text_content[max(0, start - context_window) : min(len(text_content), end + context_window)].strip()
        if len(context_slice) > 280:
            context_slice = context_slice[:280] + "…"
        if context_slice:
            if context_slice not in entry["contexts"] and len(entry["contexts"]) < 8:
                entry["contexts"].append(context_slice)
        if len(entry["bounds"]) < 500:
            entry["bounds"].append((start, end))
        if context_slice:
            normalized_context = " ".join(context_slice.split())
            groups = entry["context_groups"]
            group = groups.setdefault(
                normalized_context,
                {
                    "text": context_slice,
                    "count": 0,
                    "bounds": [],
                },
            )
            group["count"] += 1
            if len(group["bounds"]) < 10:
                group["bounds"].append({"start": start, "end": end})

    top_patterns: List[Dict[str, Any]] = []
    for signature, stats in pattern_scores.items():
        count = stats["count"]
        context_groups = stats.get("context_groups", {})
        context_variants: List[Dict[str, Any]] = []
        if context_groups:
            ordered = sorted(context_groups.values(), key=lambda item: item["count"], reverse=True)
            for variant in ordered[:5]:
                context_variants.append(
                    {
                        "text": variant["text"],
                        "count": variant["count"],
                        "bounds": variant["bounds"],
                    }
                )
        sample_contexts = [variant["text"] for variant in context_variants[:3]]
        if not sample_contexts:
            sample_contexts = stats.get("contexts", [])
        top_patterns.append(
            {
                "signature": signature,
                "count": count,
                "avg_coherence": round(stats["sum_coherence"] / count, 4),
                "avg_stability": round(stats["sum_stability"] / count, 4),
                "positions": stats["positions"][:5],
                "spans": stats["spans"][:5],
                "sample_snippet": stats["snippets"][0] if stats["snippets"] else "",
                "sample_contexts": sample_contexts,
                "context_variants": context_variants,
                "top_phrases": [],
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
            end_pos = word_positions[i + n - 1] + len(words[i + n - 1])
            stats = phrase_stats.setdefault(
                phrase,
                {
                    "count": 0,
                    "positions": [],
                    "bounds": [],
                    "occurrences": [],
                    "display": None,
                },
            )
            stats["count"] += 1
            stats["positions"].append(start_pos)
            stats["bounds"].append((start_pos, end_pos))
            snippet_original = text_content[start_pos:end_pos]
            if stats["display"] is None:
                stats["display"] = snippet_original.strip()
            if len(stats["occurrences"]) < 8:
                context_radius = 80
                context_start = max(0, start_pos - context_radius)
                context_end = min(len(text_content), end_pos + context_radius)
                context_slice = text_content[context_start:context_end].strip()
                if len(context_slice) > 220:
                    context_slice = context_slice[:220] + "…"
                stats["occurrences"].append(
                    {
                        "start": start_pos,
                        "end": end_pos,
                        "snippet": snippet_original.strip(),
                        "context": context_slice,
                    }
                )

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
        bounds_sorted = sorted(stats["bounds"], key=lambda item: item[0])
        occurrences_preview = stats["occurrences"]
        variant_map: Dict[str, Dict[str, Any]] = {}
        for occurrence in occurrences_preview:
            context_text = (occurrence.get("context") or occurrence.get("snippet") or "").strip()
            if not context_text:
                continue
            normalized = " ".join(context_text.split())
            record = variant_map.setdefault(
                normalized,
                {
                    "text": context_text,
                    "count": 0,
                    "bounds": [],
                },
            )
            record["count"] += 1
            if len(record["bounds"]) < 10:
                record["bounds"].append({"start": occurrence.get("start", 0), "end": occurrence.get("end", occurrence.get("start", 0))})
        variants = sorted(variant_map.values(), key=lambda item: item["count"], reverse=True)
        display_text = (stats.get("display") or phrase).strip()
        if not display_text or display_text.lower() in {"undefined", "null", "none"}:
            if variants:
                display_text = variants[0]["text"]
        display_text = display_text or phrase
        repeating_sequences.append(
            {
                "phrase": display_text,
                "normalized_phrase": phrase,
                "base_phrase": phrase,
                "frequency": freq,
                "periodicity": periodicity,
                "positions": positions[:25],
                "bounds": [{"start": start, "end": end} for start, end in bounds_sorted[:25]],
                "examples": occurrences_preview[:5],
                "variants": variants[:5],
            }
        )

    repeating_sequences.sort(key=lambda item: item["frequency"], reverse=True)
    if not repeating_sequences:
        # Fallback to single tokens if no phrases repeat
        for token, stats in token_stats.items():
            if stats["count"] < 2:
                continue
            positions = sorted(stats["positions"])
            variant_map: Dict[str, Dict[str, Any]] = {}
            for start in positions[:25]:
                snippet = token
                context_start = max(0, start - 80)
                context_end = min(len(text_content), start + len(token) + 80)
                context_slice = text_content[context_start:context_end].strip()
                if len(context_slice) > 220:
                    context_slice = context_slice[:220] + "…"
                normalized_context = " ".join(context_slice.split()) or snippet
                record = variant_map.setdefault(
                    normalized_context,
                    {
                        "text": context_slice or snippet,
                        "count": 0,
                        "bounds": [],
                    },
                )
                record["count"] += 1
                if len(record["bounds"]) < 10:
                    record["bounds"].append({"start": start, "end": start + len(token)})
            variants = sorted(variant_map.values(), key=lambda item: item["count"], reverse=True)
            repeating_sequences.append(
                {
                    "phrase": token,
                    "normalized_phrase": token.lower(),
                    "base_phrase": token,
                    "frequency": stats["count"],
                    "periodicity": None,
                    "positions": positions[:25],
                    "bounds": [
                        {
                            "start": start,
                            "end": start + len(token),
                        }
                        for start in positions[:25]
                    ],
                    "examples": [
                        {
                            "start": start,
                            "end": start + len(token),
                            "snippet": token,
                            "context": text_content[max(0, start - 40) : min(len(text_content), start + len(token) + 40)].strip(),
                        }
                        for start in positions[:5]
                    ],
                    "variants": variants[:5],
                }
            )
        repeating_sequences.sort(key=lambda item: item["frequency"], reverse=True)
    repeating_sequences = repeating_sequences[:10]

    top_signature_set = {item["signature"] for item in top_patterns}
    pattern_bounds_lookup: Dict[str, List[tuple[int, int]]] = {
        signature: pattern_scores.get(signature, {}).get("bounds", [])
        for signature in top_signature_set
    }
    pattern_phrase_links: Dict[str, Counter[str]] = {
        signature: Counter() for signature in top_signature_set
    }

    for sequence in repeating_sequences:
        normalized = sequence.get("normalized_phrase")
        stats = phrase_stats.get(normalized) if normalized else None
        related: Counter[str] = Counter()
        if stats and stats.get("bounds"):
            intervals = stats["bounds"]
        else:
            intervals = []
            for bound in sequence.get("bounds") or []:
                if isinstance(bound, dict):
                    start = int(bound.get("start", 0))
                    end = int(bound.get("end", 0))
                    intervals.append((start, end))
        for start, end in intervals:
            for signature, span_bounds in pattern_bounds_lookup.items():
                for span_start, span_end in span_bounds:
                    if start < span_end and end > span_start:
                        related[signature] += 1
                        break
        sequence["related_signatures"] = [
            {"signature": signature, "hits": count}
            for signature, count in related.most_common(5)
        ]
        for signature, count in related.items():
            counter = pattern_phrase_links.get(signature)
            if counter is not None and normalized:
                counter[normalized] += count

    pattern_lookup = {item["signature"]: item for item in top_patterns}
    for signature, counter in pattern_phrase_links.items():
        pattern = pattern_lookup.get(signature)
        if not pattern:
            continue
        phrases_summary = []
        for normalized_phrase, count in counter.most_common(5):
            display_source = phrase_stats.get(normalized_phrase, {}).get("display")
            display_phrase = (display_source or normalized_phrase).strip() or normalized_phrase
            phrases_summary.append(
                {
                    "phrase": display_phrase,
                    "normalized_phrase": normalized_phrase,
                    "hits": count,
                }
            )
        pattern["top_phrases"] = phrases_summary

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


@app.post("/api/export/report")
async def export_analysis_report(request: Request) -> StreamingResponse:
    """Generate a PDF report for previously computed analysis results."""

    try:
        payload = await request.json()
    except Exception as exc:  # pragma: no cover - guard malformed JSON
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {exc}") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request payload must be an object")

    results = payload.get("results")
    if not isinstance(results, dict):
        raise HTTPException(status_code=400, detail="Missing analysis results to export")

    text_content = str(payload.get("text") or "")
    preview = text_content[:800]

    metrics = results.get("metrics") or {}
    patterns = results.get("structural_patterns") or []
    sequences = results.get("repeating_sequences") or []
    insights = results.get("interpretation") or []

    def _fmt_number(value: Any) -> str:
        try:
            return f"{int(value):,}"
        except (TypeError, ValueError):
            return "—"

    def _fmt_percent(value: Any) -> str:
        try:
            return f"{float(value) * 100:.1f}%"
        except (TypeError, ValueError):
            return "—"

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
        title="Structural Intelligence Analysis Report",
    )

    story: List[Any] = []
    story.append(Paragraph("Structural Intelligence Analysis Report", TITLE_STYLE))
    story.append(
        Paragraph(
            f"Generated: {datetime.utcnow().isoformat(timespec='seconds')}Z",
            DATE_STYLE,
        )
    )

    summary_data = [
        ["Metric", "Value"],
        ["Total Tokens", _fmt_number(metrics.get("total_tokens"))],
        ["Unique Tokens", _fmt_number(metrics.get("unique_tokens"))],
        ["Structural Coverage", _fmt_percent(metrics.get("structural_coverage"))],
        ["Repetition Ratio", _fmt_percent(metrics.get("repetition_ratio"))],
        ["Structural Signatures", _fmt_number(metrics.get("structural_signatures"))],
    ]
    summary_table = Table(summary_data, hAlign="LEFT", colWidths=[170, 200])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e2233")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#0d101c"), colors.HexColor("#13172b")]),
                ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#e6e9ff")),
                ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#2a3050")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 12))

    if patterns:
        story.append(Paragraph("Structural Signatures", HEADING_STYLE))
        for pattern in patterns[:5]:
            signature = escape(str(pattern.get("signature", "")))
            count = _fmt_number(pattern.get("count"))
            try:
                coherence = f"{float(pattern.get('avg_coherence', 0.0)):.3f}"
            except (TypeError, ValueError):
                coherence = "—"
            try:
                stability = f"{float(pattern.get('avg_stability', 0.0)):.3f}"
            except (TypeError, ValueError):
                stability = "—"
            line = f"&bull; {signature} — {count} hits · coh {coherence} · stab {stability}"
            story.append(Paragraph(line, BODY_STYLE))
            snippet = pattern.get("sample_snippet")
            if snippet:
                story.append(Paragraph(f"“{escape(str(snippet))}”", QUOTE_STYLE))
            context_variants = pattern.get("context_variants") or []
            for variant in context_variants[:3]:
                story.append(
                    Paragraph(
                        f"<i>{escape(str(variant.get('text', '')))}</i>"
                        f" &nbsp;({variant.get('count', 0)} hits)",
                        QUOTE_STYLE,
                    )
                )
        story.append(Spacer(1, 8))

    if sequences:
        story.append(Paragraph("Top Repeating Phrases", HEADING_STYLE))
        for sequence in sequences[:5]:
            phrase = escape(str(sequence.get("phrase", "")))
            frequency = _fmt_number(sequence.get("frequency"))
            periodicity = sequence.get("periodicity")
            details = f"{frequency} hits"
            try:
                if periodicity:
                    details += f" · period {float(periodicity):.1f}"
            except (TypeError, ValueError):
                pass
            story.append(Paragraph(f"&bull; {phrase} — {details}", BODY_STYLE))
            variants = sequence.get("variants") or []
            for variant in variants[:3]:
                story.append(
                    Paragraph(
                        f"<i>{escape(str(variant.get('text', '')))}</i>"
                        f" &nbsp;({variant.get('count', 0)} hits)",
                        QUOTE_STYLE,
                    )
                )
        story.append(Spacer(1, 8))

    if insights:
        story.append(Paragraph("Insights", HEADING_STYLE))
        for insight in insights[:6]:
            story.append(Paragraph(f"&bull; {escape(str(insight))}", BODY_STYLE))
        story.append(Spacer(1, 8))

    if preview:
        story.append(Paragraph("Source Preview", HEADING_STYLE))
        preview_block = escape(preview).replace("\n", "<br/>")
        if len(text_content) > len(preview):
            preview_block += "<br/><i>…truncated for brevity…</i>"
        story.append(Paragraph(preview_block, BODY_STYLE))

    if not story:
        raise HTTPException(status_code=400, detail="No content available for report generation")

    doc.build(story)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=structural-analysis-report.pdf"},
    )


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
