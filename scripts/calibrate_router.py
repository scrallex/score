#!/usr/bin/env python3
"""Calibrate router thresholds with coverage guardrails."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

TARGET_LOW = 0.05
TARGET_HIGH = 0.20


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


def compute_percentiles(values: np.ndarray, quantiles: Iterable[float]) -> Dict[str, float]:
    return {f"P{int(q * 100):02d}": float(np.nanquantile(values, q)) for q in quantiles}


def coverage_for_thresholds(
    metrics: np.ndarray,
    min_coh: float | None,
    max_ent: float | None,
    min_stab: float | None,
) -> float:
    mask = np.ones(len(metrics), dtype=bool)
    if min_coh is not None:
        mask &= metrics[:, 0] >= min_coh
    if max_ent is not None:
        mask &= metrics[:, 1] <= max_ent
    if min_stab is not None:
        mask &= metrics[:, 2] >= min_stab
    if mask.size == 0:
        return 0.0
    return float(mask.sum() / mask.size)


def choose_thresholds(metrics: np.ndarray) -> Tuple[Dict[str, float], Dict[str, int | None], float]:
    coherence = metrics[:, 0]
    entropy = metrics[:, 1]
    stability = metrics[:, 2]

    coh_qs = [0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
    ent_qs = [0.35, 0.40, 0.45, 0.50, 0.55]
    stab_qs: list[float | None] = [0.60, None, 0.55, 0.50]

    in_range: list[Tuple[Dict[str, float], Dict[str, int | None], float]] = []
    candidates: list[Tuple[Dict[str, float], Dict[str, int | None], float]] = []

    for stab_q in stab_qs:
        stab_threshold = float(np.nanquantile(stability, stab_q)) if isinstance(stab_q, float) else 0.0
        stab_pct = int(stab_q * 100) if isinstance(stab_q, float) else None
        for coh_q in coh_qs:
            coh_threshold = float(np.nanquantile(coherence, coh_q))
            coh_pct = int(coh_q * 100)
            for ent_q in ent_qs:
                ent_threshold = float(np.nanquantile(entropy, ent_q))
                ent_pct = int(ent_q * 100)
                coverage = coverage_for_thresholds(metrics, coh_threshold, ent_threshold, stab_threshold if stab_pct is not None else None)
                thresholds = {
                    "min_coh": coh_threshold,
                    "max_ent": ent_threshold,
                    "min_stab": stab_threshold if stab_pct is not None else 0.0,
                }
                percentiles = {"coh": coh_pct, "ent": ent_pct, "stab": stab_pct}
                candidate = (thresholds, percentiles, coverage)
                candidates.append(candidate)
                if TARGET_LOW <= coverage <= TARGET_HIGH:
                    in_range.append(candidate)

    if in_range:
        return sorted(in_range, key=lambda item: item[2])[0]
    if not candidates:
        raise RuntimeError("Unable to derive router thresholds")

    def distance_to_range(value: float) -> float:
        if value < TARGET_LOW:
            return TARGET_LOW - value
        if value > TARGET_HIGH:
            return value - TARGET_HIGH
        return 0.0

    return min(candidates, key=lambda item: (distance_to_range(item[2]), item[2]))


def compute_configuration(metrics: np.ndarray) -> Tuple[dict, dict, float]:
    if metrics.size == 0:
        raise ValueError("No signal metrics available for calibration")

    thresholds, percentiles, coverage = choose_thresholds(metrics)
    coherence = metrics[:, 0]
    entropy = metrics[:, 1]
    stability = metrics[:, 2]

    percentile_table = {
        "coherence": compute_percentiles(coherence, [0.10, 0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]),
        "entropy": compute_percentiles(entropy, [0.10, 0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]),
        "stability": compute_percentiles(stability, [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]),
    }

    cfg = {
        "router": {
            "foreground": thresholds,
            "foreground_percentiles": percentiles,
            "coverage": coverage,
            "triggers": {
                "min_sig_qgrams": 2,
                "max_ann_dist": 0.20,
            },
        }
    }
    return cfg, percentile_table, coverage


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/calibrate_router.py <state.json>")
    state_path = Path(sys.argv[1])
    state = load_state(state_path)
    metrics = extract_metrics(state)
    cfg, percentiles, coverage = compute_configuration(metrics)

    out_path = Path("analysis/router_config.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(json.dumps(cfg, indent=2))

    coverage_path = out_path.with_suffix(".coverage.json")
    coverage_payload = {"coverage": coverage, "percentiles": percentiles}
    coverage_path.write_text(json.dumps(coverage_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
