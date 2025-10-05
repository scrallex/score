#!/usr/bin/env python3
"""Condense raw attention heatmap PNGs into aggregated figures."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from PIL import Image

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit("matplotlib is required; install it with 'pip install matplotlib'") from exc


def discover_attention_dirs(root: Path) -> List[Path]:
    pattern = root.glob("*_transformer/attention_*")
    dirs = [path for path in pattern if path.is_dir()]
    return sorted(dirs)


def load_png_stack(png_paths: Sequence[Path]) -> np.ndarray:
    arrays: List[np.ndarray] = []
    for png_path in png_paths:
        with Image.open(png_path) as img:
            data = np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
        arrays.append(data)
    if not arrays:
        raise ValueError("No PNG files provided for aggregation")
    return np.stack(arrays, axis=0)


def aggregate_directory(directory: Path, output_dir: Path) -> Path:
    png_paths = sorted(directory.glob("*.png"))
    if not png_paths:
        raise ValueError(f"No PNG files found in {directory}")

    stack = load_png_stack(png_paths)
    mean_image = stack.mean(axis=0)
    mean_line = mean_image[:, :, :3].mean(axis=0).mean(axis=1)

    pack_name = directory.parent.name
    timestamp = directory.name.split("_")[-1]

    fig, (ax_heatmap, ax_line) = plt.subplots(2, 1, figsize=(10, 4), gridspec_kw={"height_ratios": [3, 1]})

    ax_heatmap.imshow(mean_image)
    ax_heatmap.set_title(f"{pack_name} attention mean (n={len(png_paths)})")
    ax_heatmap.set_xticks([])
    ax_heatmap.set_yticks([])

    ax_line.plot(mean_line, color="#2c7fb8")
    ax_line.set_ylabel("Mean intensity")
    ax_line.set_xlabel("Position")
    ax_line.grid(True, linestyle=":", linewidth=0.5)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"attention_{pack_name}_{timestamp}.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def update_attention_log(log_path: Path, directory: Path, aggregate_path: Path) -> None:
    if not log_path.exists():
        return
    lines = log_path.read_text(encoding="utf-8").splitlines()
    updated: List[str] = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) < 3:
            updated.append(line)
            continue
        timestamp, base_dir, version_dir = parts[:3]
        extras = parts[3:]
        if Path(version_dir) == directory:
            extras = [token for token in extras if not token.startswith("aggregate=")]
            extras.append(f"aggregate={aggregate_path}")
            updated.append("\t".join([timestamp, base_dir, version_dir, *extras]))
        else:
            updated.append(line)
    log_path.write_text("\n".join(updated) + ("\n" if updated else ""), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/eval"),
        help="Root directory containing *_transformer/attention_* folders",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures"),
        help="Directory to write aggregated figures",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("results/attention_logs.txt"),
        help="Attention log file to update with aggregate paths",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Optional explicit list of attention directories to aggregate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directories: Iterable[Path]
    if args.inputs:
        directories = args.inputs
    else:
        directories = discover_attention_dirs(args.root)

    for directory in directories:
        aggregate_path = aggregate_directory(directory, args.output)
        update_attention_log(args.log, directory, aggregate_path)
        print(f"Aggregated {directory} -> {aggregate_path}")


if __name__ == "__main__":
    main()
