"""Namespace package shim for test fixtures."""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

_pkg_root = Path(__file__).resolve().parent
_src_pkg = _pkg_root / "src" / __name__
if _src_pkg.exists():
    path_str = str(_src_pkg)
    if path_str not in __path__:
        __path__.append(path_str)
