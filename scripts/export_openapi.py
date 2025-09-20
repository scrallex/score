#!/usr/bin/env python3
"""Export the FastAPI OpenAPI schema to docs/api."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.openapi.utils import get_openapi

from stm_demo.api import app


def main() -> None:
    schema = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
        description=app.docs_url or "STM Demo API",
    )

    output_dir = Path(__file__).resolve().parents[1] / "docs" / "api"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "openapi.json"
    json_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        yaml = None  # pragma: no cover

    if yaml is not None:
        yaml_path = output_dir / "openapi.yaml"
        yaml_path.write_text(yaml.safe_dump(schema, sort_keys=False), encoding="utf-8")

    print(f"OpenAPI schema written to {json_path}")


if __name__ == "__main__":
    main()
