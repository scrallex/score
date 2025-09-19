#!/usr/bin/env python3
"""Calibrate router thresholds based on manifold signal quantiles."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def load_state(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"State file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_metrics(state: dict) -> np.ndarray:
    signals = state.get("signals") or state.get("windows")
    if not signals:
        raise ValueError("State file does not contain 'signals'. Re-run ingest with --store-signals enabled.")
    rows = []
    for sig in signals:
        metrics = sig.get("metrics", {})
        rows.append(
            (
                float(metrics.get("coherence", 0.0)),
                float(metrics.get("entropy", 1.0)),
                float(metrics.get("stability", 0.0)),
            )
        )
    return np.asarray(rows, dtype=np.float64)


def compute_quantiles(data: np.ndarray) -> dict:
    if data.size == 0:
        raise ValueError("No signal metrics available for calibration")
    coherence = data[:, 0]
    entropy = data[:, 1]
    stability = data[:, 2]
    cfg = {
        "router": {
            "foreground": {
                "min_coh": float(np.nanquantile(coherence, 0.75)),
                "max_ent": float(np.nanquantile(entropy, 0.35)),
                "min_stab": float(np.nanquantile(stability, 0.60)),
            },
            "triggers": {
                "min_sig_qgrams": 2,
                "max_ann_dist": 0.20,
            },
        }
    }
    return cfg


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/calibrate_router.py <state.json>")
    state_path = Path(sys.argv[1])
    state = load_state(state_path)
    metrics = extract_metrics(state)
    cfg = compute_quantiles(metrics)
    out_path = Path("analysis/router_config.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
