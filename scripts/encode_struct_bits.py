#!/usr/bin/env python3
"""Encode structural bit features for telemetry CSV files."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def encode_bits(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    outputs = []
    for column in columns:
        series = frame[column].astype(float)
        values = series.values
        if values.size == 0:
            continue
        dx = np.diff(values, prepend=values[0])
        ddx = np.diff(dx, prepend=dx[0])
        rolling_med = series.rolling(window=61, min_periods=1, center=True).median().to_numpy()
        deviation = (series - rolling_med).abs()
        dev_med = deviation.rolling(window=61, min_periods=1, center=True).median().to_numpy()
        ddev = np.diff(dev_med, prepend=dev_med[0])
        rolling_std = series.rolling(window=301, min_periods=1, center=True).std().fillna(0.0).to_numpy()
        zscore = (values - rolling_med) / (rolling_std + 1e-12)

        mask_up = dx > 0
        mask_accel = ddx > 0
        mask_range = ddev > 0
        mask_zpos = zscore > 0
        columns_map = {
            f"{column}__UP": mask_up,
            f"{column}__ACCEL": mask_accel,
            f"{column}__RANGEEXP": mask_range,
            f"{column}__ZPOS": mask_zpos,
        }
        data = {
            name: np.where(mask, name, "")
            for name, mask in columns_map.items()
        }
        outputs.append(pd.DataFrame(data, index=frame.index))
    if not outputs:
        return pd.DataFrame(index=frame.index)
    return pd.concat(outputs, axis=1)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/encode_struct_bits.py <csv_file>")
    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    frame = pd.read_csv(csv_path, parse_dates=["time"]).set_index("time")
    numeric_columns = [col for col in frame.columns if frame[col].dtype.kind in {"f", "i", "u"}]
    if not numeric_columns:
        print(f"No numeric columns found in {csv_path}")
        return
    bits = encode_bits(frame, numeric_columns)
    output_path = csv_path.with_name(csv_path.stem + "_bits.csv")
    bits.to_csv(output_path)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
