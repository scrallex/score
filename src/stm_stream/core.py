"""Stub streaming runtime entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from .router import RouterConfig, StreamingRouter


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
    guardrail = cfg.get("router", {}).get("guardrail", {})
    router = StreamingRouter(
        RouterConfig(
            coverage_min=float(guardrail.get("coverage_min", 0.05)),
            coverage_max=float(guardrail.get("coverage_max", 0.20)),
        )
    )
    print("STM streaming runtime not fully implemented yet.")
    print(f"Loaded config: {args.config}")
    print(f"Router guardrail: {router.config}")
