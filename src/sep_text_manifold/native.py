"""Helpers for optional native (C++) accelerated routines.

This module wraps the :mod:`sep_quantum` pybind11 bindings so callers in the
STM codebase can consume the full manifold metrics whenever the native kernel
is available.  Each helper raises ``RuntimeError`` if the extension has not
been built, allowing callers to fall back to the pure-Python implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Sequence

try:  # pragma: no cover - optional dependency
    from sep_quantum import (  # type: ignore
        QFHAggregateEvent as _QFHAggregateEvent,
        QFHEvent as _QFHEvent,
        QFHResult as _QFHResult,
        QFHState as _QFHState,
        aggregate_events as _native_aggregate_events,
        analyze_bits as _native_analyze_bits,
        analyze_window as _native_analyze_window,
        transform_rich as _native_transform_rich,
    )
    HAVE_NATIVE = True
except ImportError:  # pragma: no cover - optional dependency
    _native_aggregate_events = None
    _native_analyze_bits = None
    _native_analyze_window = None
    _native_transform_rich = None
    _QFHAggregateEvent = None
    _QFHEvent = None
    _QFHResult = None
    _QFHState = None
    HAVE_NATIVE = False

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sep_quantum import QFHAggregateEvent, QFHEvent, QFHResult, QFHState
else:
    QFHAggregateEvent = _QFHAggregateEvent  # type: ignore
    QFHEvent = _QFHEvent  # type: ignore
    QFHResult = _QFHResult  # type: ignore
    QFHState = _QFHState  # type: ignore

__all__ = [
    "HAVE_NATIVE",
    "QFHAggregateEvent",
    "QFHEvent",
    "QFHResult",
    "QFHState",
    "aggregate_events",
    "analyze_bits",
    "analyze_window",
    "bits_from_bytes",
    "set_use_native",
    "transform_rich",
    "use_native",
]


_USE_NATIVE = False


def _require_native(name: str) -> None:
    if not HAVE_NATIVE:
        raise RuntimeError("Native quantum bindings are not available")


def analyze_bits(bits: Sequence[int]) -> dict[str, float]:
    """Return aggregate metrics for ``bits`` using the native kernel."""

    _require_native("analyze_bits")
    if _native_analyze_bits is None:  # pragma: no cover - defensive guard
        raise RuntimeError("Native build missing analyze_bits")
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


def analyze_window(bits: Sequence[int]):
    """Return the full :class:`sep_quantum.QFHResult` for ``bits``."""

    _require_native("analyze_window")
    if _native_analyze_window is None:
        raise RuntimeError("Native build missing analyze_window")
    return _native_analyze_window(list(bits))


def transform_rich(bits: Sequence[int]):  # type: ignore[override]
    """Return the sequence of :class:`QFHEvent` instances for ``bits``."""

    _require_native("transform_rich")
    if _native_transform_rich is None:
        raise RuntimeError("Native build missing transform_rich")
    return list(_native_transform_rich(list(bits)))


def aggregate_events(events):  # type: ignore[override]
    """Aggregate events into :class:`QFHAggregateEvent` spans."""

    _require_native("aggregate_events")
    if _native_aggregate_events is None:
        raise RuntimeError("Native build missing aggregate_events")
    return list(_native_aggregate_events(list(events)))


def bits_from_bytes(data: bytes) -> Iterable[int]:
    for byte in data:
        for shift in range(7, -1, -1):
            yield (byte >> shift) & 1


def set_use_native(enabled: bool) -> None:
    """Toggle whether high-level helpers should prefer the native kernel."""

    global _USE_NATIVE
    _USE_NATIVE = bool(enabled) and HAVE_NATIVE


def use_native() -> bool:
    """Return ``True`` when the native engine should be used by defaults."""

    return HAVE_NATIVE and _USE_NATIVE
