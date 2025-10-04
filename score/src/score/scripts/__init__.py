"""Shim implementations required by unit tests.

This package proxies a subset of the real ``scripts`` CLI modules so the
test suite can import them via ``score.scripts`` (matching the layout in
the upstream repositories).  Where we only need lightweight behaviour we
ship small local stubs, e.g. ``logistics_guardrail_demo``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from . import logistics_guardrail_demo

_ROOT = Path(__file__).resolve().parents[4]
_SCRIPT_ROOT = _ROOT / "scripts"


def _load_script(name: str) -> ModuleType:
    module_name = f"{__name__}.{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    script_path = _SCRIPT_ROOT / f"{name}.py"
    if not script_path.exists():  # pragma: no cover - defensive; tests cover expected names
        raise ImportError(f"Script '{name}' not found at {script_path}")
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:  # pragma: no cover - loader creation failure
        raise ImportError(f"Unable to load script '{name}'")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


planbench_to_stm = _load_script("planbench_to_stm")
enrich_features = _load_script("enrich_features")
calibrate_router = _load_script("calibrate_router")

__all__ = [
    "logistics_guardrail_demo",
    "planbench_to_stm",
    "enrich_features",
    "calibrate_router",
]
