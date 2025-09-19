"""Adapter utilities for THEMIS mission telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd


@dataclass
class ThemisConfig:
    channels: List[str]
    resample: str = "1S"
    timezone: str = "UTC"


DEFAULT_CONFIG = ThemisConfig(
    channels=[
        "thg_mag_x",
        "thg_mag_y",
        "thg_mag_z",
        "thg_plasma_density",
    ]
)


def load_raw(csv_path: Path, config: ThemisConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Load a THEMIS CSV into a DataFrame ({time, channels})."""
    df = pd.read_csv(csv_path, parse_dates=["time"])
    cols = ["time"] + config.channels
    return df[cols]


def resample_bits(df: pd.DataFrame, config: ThemisConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Placeholder: map THEMIS numeric channels to structural bit features."""
    raise NotImplementedError("BIT encoding for THEMIS not yet implemented")
