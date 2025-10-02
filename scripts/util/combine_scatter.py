#!/usr/bin/env python3
"""Combine two scatter plot PNGs into a single side-by-side image."""

from __future__ import annotations

import sys
from pathlib import Path
from PIL import Image


def combine(left_path: Path, right_path: Path, output_path: Path) -> None:
    if not left_path.exists() or not right_path.exists():
        return
    left = Image.open(left_path)
    right = Image.open(right_path)
    canvas = Image.new(
        "RGB",
        (left.width + right.width, max(left.height, right.height)),
        "white",
    )
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: combine_scatter.py left.png right.png output.png", file=sys.stderr)
        return 1
    left, right, output = map(Path, argv)
    combine(left, right, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
