"""Candle ingestion helpers for STM backtesting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


CANDLE_FIELD_ALIASES = {
    "open": {"open", "o", "Open", "OPEN"},
    "high": {"high", "h", "High", "HIGH"},
    "low": {"low", "l", "Low", "LOW"},
    "close": {"close", "c", "Close", "CLOSE"},
    "volume": {"volume", "v", "Volume", "VOL", "VOLUME"},
}

TIMESTAMP_ALIASES = [
    "timestamp",
    "ts",
    "time",
    "datetime",
    "date",
    "Timestamp",
    "Time",
]


@dataclass(slots=True)
class CandleRecord:
    """Normalized candle record used by the STM manifold builder."""

    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


def _resolve_column(columns: Iterable[str], aliases: Iterable[str]) -> Optional[str]:
    cols = list(columns)
    for alias in aliases:
        if alias in cols:
            return alias
    lower_map = {c.lower(): c for c in cols}
    for alias in aliases:
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    return None


def load_candles_csv(
    path: Path | str,
    *,
    timestamp_col: str | None = None,
    tz: str | timezone | None = timezone.utc,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load OHLCV candles from a CSV file and return a normalized DataFrame.

    Parameters
    ----------
    path:
        CSV file containing candle data.
    timestamp_col:
        Optional explicit timestamp column. If not provided, the loader tries
        common aliases (``timestamp``, ``time``, ``datetime``...).
    tz:
        Timezone for naive timestamps. Defaults to UTC.
    start, end:
        Optional inclusive filtering bounds.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    ts_col = timestamp_col or _resolve_column(df.columns, TIMESTAMP_ALIASES)
    if ts_col is None:
        raise ValueError(
            f"No timestamp column found in {csv_path}. Add a header or pass timestamp_col explicitly."
        )

    column_map: dict[str, str] = {}
    for key, aliases in CANDLE_FIELD_ALIASES.items():
        col = _resolve_column(df.columns, aliases)
        if col is None:
            raise ValueError(f"Missing required field '{key}' in {csv_path}")
        column_map[key] = col

    ts = pd.to_datetime(df[ts_col], utc=False, errors="coerce")
    if ts.isnull().any():
        raise ValueError(f"Timestamp column '{ts_col}' in {csv_path} contains unparsable values")

    target_tz = tz or timezone.utc
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(target_tz, ambiguous="infer", nonexistent="shift_forward")
    else:
        ts = ts.dt.tz_convert(target_tz)
    ts = ts.dt.tz_convert(timezone.utc)

    candles = pd.DataFrame(
        {
            "timestamp": ts,
            "open": pd.to_numeric(df[column_map["open"]], errors="coerce"),
            "high": pd.to_numeric(df[column_map["high"]], errors="coerce"),
            "low": pd.to_numeric(df[column_map["low"]], errors="coerce"),
            "close": pd.to_numeric(df[column_map["close"]], errors="coerce"),
            "volume": pd.to_numeric(df[column_map["volume"]], errors="coerce").fillna(0.0),
        }
    )

    candles = candles.dropna(subset=["open", "high", "low", "close"]).copy()
    candles.sort_values("timestamp", inplace=True)
    candles.reset_index(drop=True, inplace=True)

    if start is not None:
        candles = candles[candles["timestamp"] >= pd.to_datetime(start, utc=True)]
    if end is not None:
        candles = candles[candles["timestamp"] <= pd.to_datetime(end, utc=True)]

    candles["timestamp_ms"] = (candles["timestamp"].astype("int64") // 1_000_000)
    return candles


def preprocess_candles(df: pd.DataFrame, *, enforce_m1: bool = True) -> pd.DataFrame:
    """Clean and validate a raw candle DataFrame.

    The function ensures monotonic timestamps, removes duplicates, and optionally
    checks that the cadence corresponds to one-minute bars.
    """
    if df.empty:
        return df

    data = df.copy()
    data = data.drop_duplicates(subset="timestamp_ms")
    data.sort_values("timestamp_ms", inplace=True)
    data.reset_index(drop=True, inplace=True)

    if enforce_m1 and len(data) >= 2:
        deltas = data["timestamp_ms"].diff().dropna()
        # 60 seconds ± 2 seconds tolerance
        mismatch = deltas[(deltas < 58_000) | (deltas > 62_000)]
        if not mismatch.empty:
            raise ValueError(
                "Data cadence is not M1 within tolerance; found sample deltas outside 58-62s range"
            )

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        if col != "volume" and data[col].isnull().any():
            raise ValueError(f"Column '{col}' contains non-numeric values after conversion")
    data["volume"] = data["volume"].fillna(0.0)

    # Guard against inverted candles
    invalid = data[(data["high"] < data["low"]) | (data["high"] < data["open"]) | (data["high"] < data["close"]) |
                   (data["low"] > data["open"]) | (data["low"] > data["close"]) | (data["open"] <= 0) | (data["close"] <= 0)]
    if not invalid.empty:
        raise ValueError("Found malformed OHLC rows (negative or inconsistent bounds)")

    # Forward-fill zero volume if the series is sparse, otherwise keep zeros.
    if (data["volume"] == 0).all():
        data["volume"] = data["volume"].astype(float)
    else:
        data["volume"] = data["volume"].replace({math.nan: 0.0})

    return data
