"""FastAPI service exposing canned STM demo payloads."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException

from demo.standalone import DEFAULT_OUTPUT, build_payload

app = FastAPI(title="STM Demo API", version="0.1.0")

_output_path = Path(os.environ.get("STM_DEMO_PAYLOAD", str(DEFAULT_OUTPUT))).resolve()
_payload_lock = threading.Lock()
_cached_payload: Dict[str, Any] = {}
_cached_mtime: float = 0.0


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


@app.on_event("startup")
def _on_startup() -> None:
    _load_payload(force=True)


@app.get("/health")
def health() -> Dict[str, Any]:
    payload = _load_payload()
    return {"status": "ok", "generated_at": payload.get("generated_at")}


@app.get("/api/demo")
def get_demo_payload() -> Dict[str, Any]:
    return _load_payload()


@app.get("/api/demo/list")
def list_demos() -> Dict[str, Any]:
    payload = _load_payload()
    demos = payload.get("demos", {})
    return {"keys": sorted(demos.keys())}


@app.get("/api/demo/{demo_id}")
def get_demo(demo_id: str) -> Dict[str, Any]:
    payload = _load_payload()
    demos = payload.get("demos", {})
    data = demos.get(demo_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"demo '{demo_id}' not found")
    return {"id": demo_id, "data": data}


@app.post("/api/demo/refresh")
def refresh_demo() -> Dict[str, Any]:
    payload = _load_payload(force=True)
    return {"status": "refreshed", "generated_at": payload.get("generated_at")}


@app.get("/api/meta")
def get_meta() -> Dict[str, Any]:
    payload = _load_payload()
    return {
        "generated_at": payload.get("generated_at"),
        "sources": payload.get("sources", {}),
        "assets": payload.get("assets", {}),
    }


@app.get("/api/assets/{key}")
def get_asset(key: str) -> Dict[str, Any]:
    payload = _load_payload()
    assets = payload.get("assets", {})
    path = assets.get(key)
    if not path:
        raise HTTPException(status_code=404, detail=f"asset '{key}' not found")
    return {"key": key, "path": path}


@app.get("/api/ping")
def ping() -> Dict[str, str]:
    return {"status": "alive"}
