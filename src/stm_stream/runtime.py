"""Minimal streaming runtime wiring manifold logs to HTTP endpoints."""

from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable

from fastapi import FastAPI
from pydantic import BaseModel

from sep_text_manifold.stream import follow_log


class StreamingRuntime:
    """Follow a manifold log (or other source) and expose /stm endpoints."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.windows: Deque[Dict[str, Any]] = deque(maxlen=config.get("buffer_size", 5000))
        self.lock = threading.Lock()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        source = self.config.get("source", {})
        kind = source.get("kind", "manifold_log")
        if kind == "manifold_log":
            path = Path(source.get("path", "analysis/manifold.log"))
            thread = threading.Thread(target=self._run_log, args=(path,), daemon=True)
            thread.start()
            self._threads.append(thread)
        elif kind == "file":
            path = Path(source.get("path", "telemetry.ndjson"))
            thread = threading.Thread(target=self._run_file, args=(path,), daemon=True)
            thread.start()
            self._threads.append(thread)
        else:
            raise ValueError(f"Unsupported source kind '{kind}'")

    def _run_file(self, path: Path) -> None:
        """Tail a newline-delimited JSON file (very light implementation)."""
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        last_size = 0
        while True:
            if not path.exists():
                time.sleep(0.2)
                continue
            data = path.read_text(encoding="utf-8")
            if len(data) == last_size:
                time.sleep(0.2)
                continue
            lines = data.splitlines()
            for line in lines:
                if not line.strip():
                    continue
                record = json.loads(line)
                self._append_window(record)
            last_size = len(data)
            time.sleep(0.2)

    def _run_log(self, path: Path) -> None:
        if not path.exists():
            return
        for record in follow_log(path, from_start=True):
            record["timestamp"] = time.time()
            self._append_window(record)

    def _append_window(self, record: Dict[str, Any]) -> None:
        with self.lock:
            self.windows.append(record)

    def create_app(self) -> FastAPI:
        app = FastAPI(title="STM Streaming Runtime")

        class SeenRequest(BaseModel):
            signature: str
            limit: int = 20

        @app.get("/stm/health")
        def health() -> Dict[str, Any]:  # pragma: no cover - simple route
            with self.lock:
                return {"status": "ok", "buffer": len(self.windows)}

        @app.post("/stm/seen")
        def seen(req: SeenRequest) -> Dict[str, Any]:
            with self.lock:
                matches = [w for w in reversed(self.windows) if w.get("signature") == req.signature]
                return {"windows": matches[: req.limit]}

        return app
