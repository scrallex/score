"""Console entrypoint for the STM demo FastAPI service."""

from __future__ import annotations

import argparse

import uvicorn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the STM demo API server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Listen port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development only)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uvicorn.run(
        "stm_demo.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":  # pragma: no cover
    main()
