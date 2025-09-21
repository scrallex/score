#!/usr/bin/env python3
"""Calibrate router thresholds with coverage guardrails."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from run_permutation_guardrail import summarise_domain

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
    parser.add_argument(
        "--domain-root",
        type=Path,
        help="Domain export root (requires invalid/metrics for permutation evaluation)",
    )
    parser.add_argument(
        "--permutation-output",
        type=Path,
        help="Explicit path for permutation summary JSON (defaults to <output>.permutation.json)",
    )
    parser.add_argument(
        "--permutation-iterations",
        type=int,
        default=20000,
        help="Number of shuffles when evaluating guardrail significance",
    )
    parser.add_argument(
        "--dynamic-target",
        type=float,
        help="Fallback coverage target (fraction) used when permutation significance is too weak",
    )
    parser.add_argument(
        "--dynamic-window",
        type=float,
        default=0.005,
        help="Coverage window applied around --dynamic-target",
    )
    parser.add_argument(
        "--pvalue-threshold",
        type=float,
        default=0.05,
        help="Threshold for permutation metric; guardrail drops when exceeded",
    )
    parser.add_argument(
        "--pvalue-metric",
        choices=("min", "mean"),
        default="min",
        help="Permutation metric compared against --pvalue-threshold",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.001,
        help="Hard lower bound for coverage search windows",
    )
    args = parser.parse_args()

    if args.dynamic_target is not None and args.domain_root is None:
        parser.error("--domain-root is required when --dynamic-target is specified")

    state_path = args.state
    state = load_state(state_path)
    metrics = extract_metrics(state)
    base_cfg, base_percentiles, base_coverage = compute_configuration(metrics, args.target_low, args.target_high)

    out_path = args.output
    if out_path is None:
        out_path = state_path.with_name(f"{state_path.stem}_router_config.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the base configuration so permutation evaluation (if any) has access to it.
    out_path.write_text(json.dumps(base_cfg, indent=2), encoding="utf-8")

    base_summary: Optional[Dict[str, Any]] = None
    dynamic_summary: Optional[Dict[str, Any]] = None
    mode = "base"
    final_cfg = base_cfg
    final_percentiles = base_percentiles
    final_coverage = base_coverage

    permutation_output = args.permutation_output or out_path.with_suffix(".permutation.json")
    permutation_output = Path(permutation_output)
    if args.domain_root is not None:
        base_summary = summarise_domain(
            domain_root=args.domain_root,
            config_path=out_path,
            output_path=permutation_output,
            iterations=args.permutation_iterations,
        )

        metric_key = "p_value_min" if args.pvalue_metric == "min" else "p_value_mean"
        metric_value = base_summary.get(metric_key) if base_summary else None

        should_drop = False
        metric_float: Optional[float] = None
        if args.dynamic_target is not None and metric_value is not None:
            try:
                metric_float = float(metric_value)
            except (TypeError, ValueError):
                metric_float = None
            if metric_float is not None and metric_float > args.pvalue_threshold:
                should_drop = True

        if should_drop:
            dynamic_low = max(args.dynamic_target - args.dynamic_window, args.min_coverage)
            dynamic_high = args.dynamic_target + args.dynamic_window

            # Preserve the base configuration and permutation summary for audit.
            base_config_path = out_path.with_suffix(".base.json")
            base_config_path.write_text(json.dumps(base_cfg, indent=2), encoding="utf-8")
            if permutation_output.exists():
                base_perm_path = permutation_output.with_name(
                    f"{permutation_output.stem}_base{permutation_output.suffix}"
                )
                permutation_output.replace(base_perm_path)

            dynamic_cfg, dynamic_percentiles, dynamic_coverage = compute_configuration(
                metrics,
                dynamic_low,
                dynamic_high,
            )
            out_path.write_text(json.dumps(dynamic_cfg, indent=2), encoding="utf-8")
            dynamic_summary = summarise_domain(
                domain_root=args.domain_root,
                config_path=out_path,
                output_path=permutation_output,
                iterations=args.permutation_iterations,
            )
            final_cfg = dynamic_cfg
            final_percentiles = dynamic_percentiles
            final_coverage = dynamic_coverage
            mode = "dynamic"
            sys.stderr.write(
                (
                    f"[calibrate_router] Permutation {metric_key}={metric_float:.3f} > {args.pvalue_threshold:.3f}; "
                    f"recalibrated guardrail to target {args.dynamic_target:.3f} (coverage {final_coverage:.4f}).\n"
                )
            )

    print(json.dumps(final_cfg, indent=2))

    coverage_path = out_path.with_suffix(".coverage.json")
    coverage_payload: Dict[str, Any] = {
        "mode": mode,
        "coverage": final_coverage,
        "percentiles": final_percentiles,
        "base": {
            "coverage": base_coverage,
            "percentiles": base_percentiles,
            "summary": base_summary,
        },
    }
    if dynamic_summary is not None:
        coverage_payload.update(
            {
                "dynamic": {
                    "coverage": final_coverage,
                    "percentiles": final_percentiles,
                    "summary": dynamic_summary,
                    "target": args.dynamic_target,
                    "window": args.dynamic_window,
                },
                "pvalue_metric": args.pvalue_metric,
                "pvalue_threshold": args.pvalue_threshold,
            }
        )

    coverage_path.write_text(json.dumps(coverage_payload, indent=2), encoding="utf-8")
    sys.stderr.write(
        f"Calibrated coverage {final_coverage:.4f} using mode '{mode}' -> {out_path}\n"
    )


if __name__ == "__main__":
    main()
