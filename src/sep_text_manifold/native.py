"""Helpers for optional native (C++) accelerated routines."""

from __future__ import annotations

from typing import Iterable, Sequence

try:  # pragma: no cover - optional dependency
    from sep_quantum import analyze_bits as _native_analyze_bits  # type: ignore
    HAVE_NATIVE = True
except ImportError:  # pragma: no cover - optional dependency
    _native_analyze_bits = None
    HAVE_NATIVE = False


def analyze_bits(bits: Sequence[int]) -> dict[str, float]:
    if not HAVE_NATIVE or _native_analyze_bits is None:
        raise RuntimeError("Native quantum bindings are not available")
    metrics = _native_analyze_bits(list(bits))
    return {
        "coherence": float(metrics.coherence),
        "stability": float(metrics.stability),
        "entropy": float(metrics.entropy),
        "rupture": float(metrics.rupture),
        "lambda_hazard": float(metrics.lambda_hazard),
        "sig_c": int(metrics.sig_c),
        "sig_s": int(metrics.sig_s),
        "sig_e": int(metrics.sig_e),
    }


def bits_from_bytes(data: bytes) -> Iterable[int]:
    for byte in data:
        for shift in range(7, -1, -1):
            yield (byte >> shift) & 1
