"""Python STM manifold builder mirroring the SEP C++ implementation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import pandas as pd

try:  # Import lazily so unit tests can inject stubs if needed
    import sep_quantum
except ModuleNotFoundError as exc:  # pragma: no cover - surfaced as runtime error
    sep_quantum = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:  # pragma: no branch
    _IMPORT_ERROR = None


@dataclass(slots=True)
class STMManifoldBuilder:
    """Generate STM fingerprints from OHLCV candles.

    Parameters mirror the SEP `buildManifold` routine but lift the limit on the
    number of emitted signals so backtests can consume full histories.
    """

    signature_precision: int = 2
    lookback_minutes: float = 60.0
    max_signals: Optional[int] = None
    step_divisor: int = 32
    _scale: float = field(init=False, repr=False)
    _lookback_ms: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if sep_quantum is None:
            raise RuntimeError(
                "sep_quantum module not available. Build the native extension or run `pip install -e .`"
            ) from _IMPORT_ERROR
        if self.signature_precision < 0:
            raise ValueError("signature_precision must be non-negative")
        if self.lookback_minutes <= 0:
            raise ValueError("lookback_minutes must be positive")
        if self.step_divisor <= 0:
            raise ValueError("step_divisor must be positive")
        self._scale = 10.0**self.signature_precision
        self._lookback_ms = int(self.lookback_minutes * 60_000)

    # ---------------------------------------------------------------------
    # Public API

    def build(self, candles: pd.DataFrame, *, instrument: str = "UNKNOWN") -> pd.DataFrame:
        """Return STM signals for an instrument.

        Parameters
        ----------
        candles:
            Preprocessed candle DataFrame produced by ``preprocess_candles``.
        instrument:
            Label attached to every emitted signal.
        """
        if candles.empty or len(candles) < 2:
            return self._empty_frame()

        bits = self._encode_bits(candles)
        if not bits:
            return self._empty_frame()

        window = self._select_window(len(bits))
        start, step = self._iteration_params(len(bits), window)
        indices = list(range(start, len(bits) + 1, step))

        records: List[Dict[str, object]] = []
        repetition_history: Dict[str, Deque[int]] = {}

        for i in indices:
            begin = i - window
            if begin < 0:
                continue
            sub_bits = bits[begin:i]
            if not sub_bits:
                continue
            result = sep_quantum.analyze_window(sub_bits)
            coherence = float(result.coherence)
            rupture = float(result.rupture_ratio)
            entropy = float(result.entropy)
            stability = 1.0 - rupture

            ts_ms = int(candles.iloc[begin + 1]["timestamp_ms"])
            price = float(candles.iloc[begin + 1]["close"])

            signature = self._make_signature(coherence, stability, entropy)
            history = repetition_history.setdefault(signature, deque())
            while history and ts_ms > history[0] and ts_ms - history[0] > self._lookback_ms:
                history.popleft()
            history.append(ts_ms)
            count = len(history)
            first_seen = history[0] if history else ts_ms

            hazard = max(0.0, min(1.0, rupture))
            records.append(
                {
                    "instrument": instrument,
                    "timestamp": pd.to_datetime(ts_ms, unit="ms", utc=True),
                    "timestamp_ms": ts_ms,
                    "timestamp_ns": ts_ms * 1_000_000,
                    "price": price,
                    "coherence": coherence,
                    "stability": stability,
                    "entropy": entropy,
                    "rupture": rupture,
                    "lambda_hazard": hazard,
                    "signature": signature,
                    "repetition_count": count,
                    "repetition_first_seen_ms": first_seen,
                    "window_index": i,
                }
            )

        if not records:
            return self._empty_frame()
        df = pd.DataFrame.from_records(records)
        df.sort_values("timestamp_ms", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    # ------------------------------------------------------------------
    # Internal helpers

    @staticmethod
    def _empty_frame() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "instrument",
                "timestamp",
                "timestamp_ms",
                "timestamp_ns",
                "price",
                "coherence",
                "stability",
                "entropy",
                "rupture",
                "lambda_hazard",
                "signature",
                "repetition_count",
                "repetition_first_seen_ms",
                "window_index",
            ]
        )

    @staticmethod
    def _encode_bits(candles: pd.DataFrame) -> List[int]:
        bits: List[int] = []
        for idx in range(1, len(candles)):
            prev = candles.iloc[idx - 1]
            curr = candles.iloc[idx]
            price_up = float(curr["close"]) >= float(prev["close"])
            range_expanding = (float(curr["high"]) - float(curr["low"])) >= (
                float(prev["high"]) - float(prev["low"])
            )
            volume_increasing = float(curr["volume"]) >= float(prev["volume"])
            bit_value = 1 if price_up else 0
            if not range_expanding and not volume_increasing:
                bit_value = 0
            bits.append(bit_value)
        return bits

    @staticmethod
    def _select_window(bit_count: int) -> int:
        window = bit_count
        if window > 128:
            window = 128
        if window >= bit_count:
            if bit_count > 24:
                window = bit_count - min(8, bit_count // 4)
            elif bit_count > 12:
                window = 12
            elif bit_count > 8:
                window = 9
            else:
                window = bit_count
        return max(1, window)

    def _iteration_params(self, bit_count: int, window: int) -> tuple[int, int]:
        start = window
        if self.max_signals is not None and bit_count - window > self.max_signals:
            start = bit_count - self.max_signals
        start = max(window, start)
        step = max(1, window // self.step_divisor)
        return start, step

    def _make_signature(self, coherence: float, stability: float, entropy: float) -> str:
        def bucket(value: float) -> float:
            clipped = max(0.0, min(1.0, value))
            return round(clipped * self._scale) / self._scale

        return f"c{bucket(coherence):.{self.signature_precision}f}_s{bucket(stability):.{self.signature_precision}f}_e{bucket(entropy):.{self.signature_precision}f}"


def build_signals(
    candles: pd.DataFrame,
    *,
    instrument: str = "UNKNOWN",
    signature_precision: int = 2,
    lookback_minutes: float = 60.0,
    max_signals: Optional[int] = None,
    step_divisor: int = 32,
) -> pd.DataFrame:
    """Convenience wrapper returning STM signals with sensible defaults."""
    builder = STMManifoldBuilder(
        signature_precision=signature_precision,
        lookback_minutes=lookback_minutes,
        max_signals=max_signals,
        step_divisor=step_divisor,
    )
    return builder.build(candles, instrument=instrument)
