"""Stub streaming runtime entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

import uvicorn

from .runtime import StreamingRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run STM streaming router")
    parser.add_argument("--config", required=True, help="YAML config path")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    runtime = StreamingRuntime(cfg)
    runtime.start()
    app = runtime.create_app()
    api_cfg = cfg.get("api", {})
    host = api_cfg.get("host", "0.0.0.0")
    port = int(api_cfg.get("port", 8000))
    uvicorn.run(app, host=host, port=port)
