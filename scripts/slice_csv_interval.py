#!/usr/bin/env python3
"""Extract a time-bounded slice from a telemetry CSV (time-indexed)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slice CSV rows between two timestamps")
    parser.add_argument("input", type=Path, help="Input CSV with a 'time' column")
    parser.add_argument("start", help="Start timestamp (inclusive, ISO8601)")
    parser.add_argument("stop", help="Stop timestamp (exclusive, ISO8601)")
    parser.add_argument("output", type=Path, help="Output CSV path")
    parser.add_argument("--append", action="store_true", help="Append to output if it exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input, parse_dates=["time"])  # type: ignore[arg-type]
    mask = (df["time"] >= args.start) & (df["time"] < args.stop)
    sliced = df.loc[mask]
    if sliced.empty:
        print(f"[slice] no rows in range for {args.input}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append and args.output.exists() else "w"
    header = not (args.append and args.output.exists())
    sliced.to_csv(args.output, index=False, mode=mode, header=header)
    print(f"[slice] wrote {len(sliced)} rows → {args.output}")


if __name__ == "__main__":
    main()
