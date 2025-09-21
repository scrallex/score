#!/usr/bin/env python3
"""Calibrate router thresholds with coverage guardrails."""

from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

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


def choose_thresholds(
    metrics: np.ndarray,
    target_low: float,
    target_high: float,
) -> Tuple[Dict[str, float], Dict[str, int | None], float]:
    coherence = metrics[:, 0]
    entropy = metrics[:, 1]
    stability = metrics[:, 2]

    coh_qs = list(np.arange(0.55, 0.99, 0.02)) + [0.99, 0.995]
    ent_qs = list(np.arange(0.02, 0.60, 0.02)) + [0.60]
    stab_qs: list[float | None] = [None] + list(np.arange(0.55, 0.90, 0.05)) + [0.90]

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
                if target_low <= coverage <= target_high:
                    in_range.append(candidate)

    if in_range:
        return sorted(in_range, key=lambda item: item[2])[0]
    if not candidates:
        raise RuntimeError("Unable to derive router thresholds")

    def distance_to_range(value: float) -> float:
        if value < target_low:
            return target_low - value
        if value > target_high:
            return value - target_high
        return 0.0

    return min(candidates, key=lambda item: (distance_to_range(item[2]), item[2]))


def compute_configuration(
    metrics: np.ndarray,
    target_low: float,
    target_high: float,
) -> Tuple[dict, dict, float]:
    if metrics.size == 0:
        raise ValueError("No signal metrics available for calibration")

    thresholds, percentiles, coverage = choose_thresholds(metrics, target_low, target_high)
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
    parser = argparse.ArgumentParser(description="Calibrate STM router guardrails")
    parser.add_argument("state", type=Path, help="Path to STM state JSON")
    parser.add_argument("--target-low", type=float, default=0.05, help="Lower bound for desired coverage")
    parser.add_argument("--target-high", type=float, default=0.20, help="Upper bound for desired coverage")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output router config path (defaults to <state>_router_config.json)",
    )
    args = parser.parse_args()

    state_path = args.state
    state = load_state(state_path)
    metrics = extract_metrics(state)
    cfg, percentiles, coverage = compute_configuration(metrics, args.target_low, args.target_high)

    out_path = args.output
    if out_path is None:
        out_path = state_path.with_name(f"{state_path.stem}_router_config.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(json.dumps(cfg, indent=2))

    coverage_path = out_path.with_suffix(".coverage.json")
    coverage_payload = {"coverage": coverage, "percentiles": percentiles}
    coverage_path.write_text(json.dumps(coverage_payload, indent=2), encoding="utf-8")
    sys.stderr.write(
        f"Calibrated coverage {coverage:.4f} with targets [{args.target_low:.2f}, {args.target_high:.2f}] -> {out_path}\n"
    )


if __name__ == "__main__":
    main()
