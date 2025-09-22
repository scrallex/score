"""Parity checks between Python STM replica and native QFH manifold output."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stm_backtest import load_candles_csv, preprocess_candles
from stm_backtest.signals import STMManifoldBuilder

DATA_DIR = Path(__file__).resolve().parent / "data"
GOLDEN_CSV = DATA_DIR / "eurusd_sample.csv"
NATIVE_TAIL_JSON = DATA_DIR / "eurusd_sample_native_tail50.json"
FLOAT_COLUMNS = ("coherence", "stability", "entropy", "lambda_hazard", "price")
ATOL = 1e-6


@pytest.fixture(scope="module")
def golden_slice() -> pd.DataFrame:
    raw = load_candles_csv(GOLDEN_CSV, tz="UTC")
    return preprocess_candles(raw)


@pytest.fixture(scope="module")
def native_tail() -> pd.DataFrame:
    payload = json.loads(NATIVE_TAIL_JSON.read_text(encoding="utf-8"))
    df = pd.DataFrame(payload)
    if df.empty:
        raise AssertionError("Native tail fixture is empty")
    return df


def test_python_manifold_matches_native_tail(golden_slice: pd.DataFrame, native_tail: pd.DataFrame) -> None:
    builder = STMManifoldBuilder(signature_precision=2, lookback_minutes=60, max_signals=None)
    python_signals = builder.build(golden_slice, instrument="EURUSD")
    assert len(python_signals) >= len(native_tail)

    python_tail = python_signals.tail(len(native_tail)).reset_index(drop=True)
    native_tail = native_tail.reset_index(drop=True)

    np.testing.assert_array_equal(python_tail["timestamp_ns"].to_numpy(dtype=np.int64), native_tail["timestamp_ns"].to_numpy(dtype=np.int64))
    np.testing.assert_array_equal(python_tail["signature"].to_numpy(dtype=object), native_tail["signature"].to_numpy(dtype=object))
    np.testing.assert_array_equal(python_tail["repetition_count"].to_numpy(dtype=np.int64), native_tail["repetition_count"].to_numpy(dtype=np.int64))
    np.testing.assert_array_equal(
        python_tail["repetition_first_seen_ms"].to_numpy(dtype=np.int64),
        native_tail["repetition_first_seen_ms"].to_numpy(dtype=np.int64),
    )

    for column in FLOAT_COLUMNS:
        np.testing.assert_allclose(
            python_tail[column].to_numpy(dtype=float),
            native_tail[column].to_numpy(dtype=float),
            rtol=0.0,
            atol=ATOL,
            err_msg=f"Mismatch in column '{column}' exceeds {ATOL}",
        )
