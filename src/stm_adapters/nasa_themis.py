"""THEMIS telemetry adapter producing structural bit features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


def _encode_struct_bits(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Vectorised encoding mirroring `scripts/encode_struct_bits.encode_bits`."""
    outputs = []
    for column in columns:
        series = frame[column].astype(float)
        if series.empty:
            continue
        values = series.to_numpy()
        dx = np.diff(values, prepend=values[0])
        ddx = np.diff(dx, prepend=dx[0])
        rolling_med = (
            pd.Series(values, index=series.index)
            .rolling(window=61, min_periods=1, center=True)
            .median()
            .to_numpy()
        )
        deviation = np.abs(values - rolling_med)
        dev_med = (
            pd.Series(deviation, index=series.index)
            .rolling(window=61, min_periods=1, center=True)
            .median()
            .to_numpy()
        )
        ddev = np.diff(dev_med, prepend=dev_med[0])
        rolling_std = (
            pd.Series(values, index=series.index)
            .rolling(window=301, min_periods=1, center=True)
            .std()
            .fillna(0.0)
            .to_numpy()
        )
        zscore = (values - rolling_med) / (rolling_std + 1e-12)

        columns_map = {
            f"{column}__UP": dx > 0,
            f"{column}__ACCEL": ddx > 0,
            f"{column}__RANGEEXP": ddev > 0,
            f"{column}__ZPOS": zscore > 0,
        }
        outputs.append(
            pd.DataFrame({name: np.where(mask, name, "") for name, mask in columns_map.items()}, index=series.index)
        )
    if not outputs:
        return pd.DataFrame(index=frame.index)
    return pd.concat(outputs, axis=1)


@dataclass
class ThemisConfig:
    channels: List[str]
    resample: str = "1S"


DEFAULT_CONFIG = ThemisConfig(
    channels=["thg_mag_x", "thg_mag_y", "thg_mag_z", "thg_plasma_density"],
)


class ThemisAdapter:
    """Minimal adapter turning THEMIS CSV telemetry into bit features."""

    def __init__(self, config: ThemisConfig = DEFAULT_CONFIG) -> None:
        self.config = config

    def load(self, csv_path: Path) -> pd.DataFrame:
        frame = pd.read_csv(csv_path, parse_dates=["time"])
        cols = ["time"] + self.config.channels
        return frame[cols].set_index("time").sort_index()

    def to_bits(self, frame: pd.DataFrame) -> pd.DataFrame:
        numeric = frame[self.config.channels].resample(self.config.resample).mean().interpolate()
        return _encode_struct_bits(numeric, self.config.channels)

    def run(self, csv_path: Path, output_dir: Path | None = None) -> Path:
        frame = self.load(csv_path)
        bits = self.to_bits(frame)
        output_dir = output_dir or csv_path.parent / f"{csv_path.stem}_stm"
        output_dir.mkdir(parents=True, exist_ok=True)

        bit_csv = output_dir / f"{csv_path.stem}_bits.csv"
        bits.reset_index().to_csv(bit_csv, index=False)

        tokens = bits.apply(lambda row: " ".join(token for token in row.values if token), axis=1)
        text_path = output_dir / f"{csv_path.stem}_tokens.txt"
        text_path.write_text("\n".join(tokens), encoding="utf-8")
        return text_path
