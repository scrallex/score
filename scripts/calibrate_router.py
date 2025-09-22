#!/usr/bin/env python3
"""Calibrate router thresholds with coverage guardrails."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


def build_quantile_sets(
    sample_size: int,
    *,
    extra_coh: Optional[Iterable[float]] = None,
    extra_ent: Optional[Iterable[float]] = None,
    extra_stab: Optional[Iterable[float]] = None,
) -> Tuple[List[float], List[float], List[float | None]]:
    coherence = list(np.arange(0.45, 0.91, 0.02))
    if sample_size > 600:
        coherence.extend(np.arange(0.91, 0.99, 0.01))
    else:
        coherence.extend([0.91, 0.93, 0.95])
    coherence.extend([0.97, 0.98, 0.99, 0.995])
    if extra_coh:
        coherence.extend(extra_coh)
    coherence = sorted({round(float(q), 4) for q in coherence})

    entropy: List[float] = []
    if sample_size > 600:
        entropy.extend(np.arange(0.01, 0.20, 0.01))
    else:
        entropy.extend(np.arange(0.02, 0.20, 0.02))
    entropy.extend(np.arange(0.20, 0.60, 0.02))
    entropy.append(0.60)
    if extra_ent:
        entropy.extend(extra_ent)
    entropy = sorted({round(float(q), 4) for q in entropy})

    stability: List[float | None] = [None]
    stability.extend(np.arange(0.45, 0.90, 0.05))
    if sample_size > 500:
        stability.extend([0.90, 0.92, 0.94])
    else:
        stability.append(0.90)
    if extra_stab:
        stability.extend(extra_stab)
    # Preserve None at head while deduplicating numeric entries.
    stability_head = [None]
    stability_tail = sorted({round(float(q), 4) for q in stability if isinstance(q, float)})
    stability = stability_head + stability_tail

    return coherence, entropy, stability


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
    *,
    extra_coh_qs: Optional[Iterable[float]] = None,
    extra_ent_qs: Optional[Iterable[float]] = None,
    extra_stab_qs: Optional[Iterable[float]] = None,
) -> Tuple[Dict[str, float], Dict[str, int | None], float]:
    coherence = metrics[:, 0]
    entropy = metrics[:, 1]
    stability = metrics[:, 2]

    coh_qs, ent_qs, stab_qs = build_quantile_sets(
        len(metrics),
        extra_coh=extra_coh_qs,
        extra_ent=extra_ent_qs,
        extra_stab=extra_stab_qs,
    )

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
    *,
    extra_coh_qs: Optional[Iterable[float]] = None,
    extra_ent_qs: Optional[Iterable[float]] = None,
    extra_stab_qs: Optional[Iterable[float]] = None,
) -> Tuple[dict, dict, float]:
    if metrics.size == 0:
        raise ValueError("No signal metrics available for calibration")

    thresholds, percentiles, coverage = choose_thresholds(
        metrics,
        target_low,
        target_high,
        extra_coh_qs=extra_coh_qs,
        extra_ent_qs=extra_ent_qs,
        extra_stab_qs=extra_stab_qs,
    )
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


def build_optimisation_centres(
    base_coverage: float,
    *,
    min_coverage: float,
    dynamic_target: float | None,
    explicit_centres: Optional[Iterable[float]],
    span: float,
    samples: int,
) -> List[float]:
    centres: set[float] = set()
    if explicit_centres:
        centres.update(float(c) for c in explicit_centres if c is not None)
    centres.add(round(max(min_coverage, base_coverage), 6))
    if dynamic_target is not None:
        centres.add(round(max(min_coverage, dynamic_target), 6))
    if span > 0 and samples > 1:
        offsets = np.linspace(-span, span, samples)
        for offset in offsets:
            centre = base_coverage + float(offset)
            if centre <= 0:
                continue
            centres.add(round(max(min_coverage, centre), 6))
    return sorted(c for c in centres if c > 0)


def optimise_permutation_guardrail(
    *,
    metrics: np.ndarray,
    args: argparse.Namespace,
    base_candidate: Dict[str, Any],
    out_path: Path,
    permutation_output: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not args.optimize_permutation or args.domain_root is None:
        base_candidate.setdefault("mode", "base")
        report_row = {
            "label": "base",
            "target": base_candidate.get("target"),
            "coverage": base_candidate.get("summary", {}).get("coverage_weighted") if base_candidate.get("summary") else None,
            "coverage_observed": base_candidate.get("coverage"),
            "p_metric": base_candidate.get("metric_value"),
            "lead_mean": base_candidate.get("summary", {}).get("lead_mean") if base_candidate.get("summary") else None,
            "fails_threshold": (
                base_candidate.get("metric_value", float("inf")) > args.pvalue_threshold
                if base_candidate.get("summary")
                else None
            ),
            "config_path": str(out_path),
            "summary_path": str(permutation_output),
        }
        return base_candidate, [report_row]

    centres = build_optimisation_centres(
        base_candidate.get("coverage", 0.0),
        min_coverage=args.min_coverage,
        dynamic_target=args.dynamic_target,
        explicit_centres=args.optimize_centers,
        span=args.optimize_span,
        samples=max(args.optimize_samples, 2),
    )

    metric_key = "p_value_min" if args.pvalue_metric == "min" else "p_value_mean"

    candidates: List[Dict[str, Any]] = []

    base_summary = base_candidate.get("summary") or {}
    base_metric_value = base_summary.get(metric_key)
    base_spec = {
        "label": "base",
        "target": base_candidate.get("coverage"),
        "coverage": base_candidate.get("coverage"),
        "coverage_weighted": base_summary.get("coverage_weighted"),
        "p_metric": float(base_metric_value) if base_metric_value is not None else None,
        "lead_mean": base_summary.get("lead_mean"),
        "fails_threshold": (
            float(base_metric_value) > args.pvalue_threshold
            if base_metric_value is not None
            else True
        ),
        "coverage_gap": abs(
            float(base_summary.get("coverage_weighted", base_candidate.get("coverage", 0.0)))
            - float(base_candidate.get("coverage", 0.0))
        ) if base_candidate.get("summary") else None,
        "cfg": base_candidate.get("cfg"),
        "percentiles": base_candidate.get("percentiles"),
        "summary": base_candidate.get("summary"),
        "coverage_observed": base_candidate.get("coverage"),
        "config_path": str(out_path),
        "summary_path": str(permutation_output),
        "mode": "base",
    }
    candidates.append(base_spec)

    evaluated_centres = {round(base_candidate.get("coverage", 0.0), 6)}

    for idx, centre in enumerate(centres, start=1):
        rounded_centre = round(centre, 6)
        if rounded_centre in evaluated_centres:
            continue
        evaluated_centres.add(rounded_centre)

        lower = max(args.min_coverage, rounded_centre - args.optimize_width)
        upper = rounded_centre + args.optimize_width
        cfg, percentiles, coverage = compute_configuration(
            metrics,
            lower,
            upper,
            extra_coh_qs=args.extra_coherence,
            extra_ent_qs=args.extra_entropy,
            extra_stab_qs=args.extra_stability,
        )

        cfg_path = out_path.with_suffix(f".opt{idx}.json")
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        summary_path = permutation_output.with_name(
            f"{permutation_output.stem}_opt{idx}{permutation_output.suffix}"
        )
        summary = summarise_domain(
            domain_root=args.domain_root,
            config_path=cfg_path,
            output_path=summary_path,
            iterations=args.permutation_iterations,
        )
        metric_value = summary.get(metric_key)
        coverage_weighted = float(summary.get("coverage_weighted", coverage))
        coverage_gap = abs(coverage_weighted - rounded_centre)
        lead_mean = summary.get("lead_mean")
        spec = {
            "label": f"candidate_{idx}",
            "target": rounded_centre,
            "coverage": coverage,
            "coverage_weighted": coverage_weighted,
            "coverage_gap": coverage_gap,
            "p_metric": float(metric_value) if metric_value is not None else None,
            "lead_mean": lead_mean,
            "fails_threshold": (
                float(metric_value) > args.pvalue_threshold
                if metric_value is not None
                else True
            ),
            "cfg": cfg,
            "percentiles": percentiles,
            "summary": summary,
            "coverage_observed": coverage,
            "config_path": str(cfg_path),
            "summary_path": str(summary_path),
        }
        candidates.append(spec)

    def ranking_key(spec: Dict[str, Any]) -> Tuple[int, float, float, float]:
        metric_raw = spec.get("p_metric")
        metric_value = float(metric_raw) if isinstance(metric_raw, (int, float)) else float("inf")
        fails = spec.get("fails_threshold", True)
        coverage_gap = float(spec.get("coverage_gap", float("inf")))
        lead_mean = spec.get("lead_mean")
        lead_score = -float(lead_mean) if isinstance(lead_mean, (int, float)) else 0.0
        return (1 if fails else 0, metric_value, coverage_gap, lead_score)

    best_spec = min(candidates, key=ranking_key)

    # Clean up unused candidate files
    for spec in candidates:
        cfg_path = Path(spec["config_path"])
        summary_path = Path(spec["summary_path"])
        if spec is best_spec:
            continue
        if cfg_path.exists() and cfg_path != out_path:
            try:
                cfg_path.unlink()
            except OSError:
                pass
        if summary_path.exists() and summary_path != permutation_output:
            try:
                summary_path.unlink()
            except OSError:
                pass

    report_rows = [
        {
            "label": spec.get("label"),
            "target": spec.get("target"),
            "coverage": spec.get("coverage_weighted"),
            "coverage_observed": spec.get("coverage"),
            "p_metric": spec.get("p_metric"),
            "lead_mean": spec.get("lead_mean"),
            "fails_threshold": spec.get("fails_threshold"),
            "config_path": spec.get("config_path"),
            "summary_path": spec.get("summary_path"),
        }
        for spec in candidates
    ]

    if best_spec is base_spec:
        base_candidate.setdefault("mode", "base")
        return base_candidate, report_rows

    base_config_path = out_path.with_suffix(".base.json")
    if not base_config_path.exists():
        base_config_path.write_text(json.dumps(base_candidate["cfg"], indent=2), encoding="utf-8")

    base_summary_path = permutation_output.with_name(
        f"{permutation_output.stem}_base{permutation_output.suffix}"
    )
    if permutation_output.exists():
        try:
            permutation_output.replace(base_summary_path)
        except OSError:
            pass

    out_path.write_text(json.dumps(best_spec["cfg"], indent=2), encoding="utf-8")

    best_summary_path = Path(best_spec["summary_path"])
    if best_summary_path.exists():
        try:
            if permutation_output.exists():
                permutation_output.unlink()
        except OSError:
            pass
        try:
            best_summary_path.replace(permutation_output)
        except OSError:
            pass

    try:
        cfg_path = Path(best_spec["config_path"])
        if cfg_path.exists() and cfg_path != out_path:
            cfg_path.unlink()
    except OSError:
        pass

    optimized_candidate = {
        "cfg": best_spec["cfg"],
        "percentiles": best_spec["percentiles"],
        "coverage": float(best_spec.get("coverage_observed", 0.0)),
        "summary": best_spec["summary"],
        "target": best_spec["target"],
        "metric_value": best_spec.get("p_metric"),
        "mode": "optimized",
    }
    return optimized_candidate, report_rows

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
    parser.add_argument(
        "--optimize-permutation",
        action="store_true",
        help="Sample nearby coverage targets and select the guardrail with the strongest permutation significance",
    )
    parser.add_argument(
        "--optimize-width",
        type=float,
        default=0.004,
        help="Coverage half-window applied around each optimisation centre",
    )
    parser.add_argument(
        "--optimize-span",
        type=float,
        default=0.01,
        help="Total span around the base coverage used to generate optimisation centres",
    )
    parser.add_argument(
        "--optimize-samples",
        type=int,
        default=5,
        help="Number of evenly spaced samples across --optimize-span (minimum 2)",
    )
    parser.add_argument(
        "--optimize-centers",
        nargs="*",
        type=float,
        default=None,
        help="Explicit coverage centres (fractions) to include during optimisation",
    )
    parser.add_argument(
        "--extra-coherence",
        nargs="*",
        type=float,
        help="Additional coherence percentiles to include (values in (0, 100) or (0, 1))",
    )
    parser.add_argument(
        "--extra-entropy",
        nargs="*",
        type=float,
        help="Additional entropy percentiles to include (values in (0, 100) or (0, 1))",
    )
    parser.add_argument(
        "--extra-stability",
        nargs="*",
        type=float,
        help="Additional stability percentiles to include (values in (0, 100) or (0, 1))",
    )
    args = parser.parse_args()

    if args.dynamic_target is not None and args.domain_root is None:
        parser.error("--domain-root is required when --dynamic-target is specified")

    def _normalize_percentiles(values: Optional[Iterable[float]]) -> Optional[List[float]]:
        if not values:
            return None
        normalized: List[float] = []
        for raw in values:
            if raw is None:
                continue
            val = float(raw)
            if val > 1.0:
                val /= 100.0
            if not 0.0 < val < 1.0:
                raise ValueError("Percentiles must lie in the open interval (0, 100) or (0, 1)")
            normalized.append(val)
        return normalized if normalized else None

    args.extra_coherence = _normalize_percentiles(args.extra_coherence)
    args.extra_entropy = _normalize_percentiles(args.extra_entropy)
    args.extra_stability = _normalize_percentiles(args.extra_stability)

    state_path = args.state
    state = load_state(state_path)
    metrics = extract_metrics(state)
    base_cfg, base_percentiles, base_coverage = compute_configuration(
        metrics,
        args.target_low,
        args.target_high,
        extra_coh_qs=args.extra_coherence,
        extra_ent_qs=args.extra_entropy,
        extra_stab_qs=args.extra_stability,
    )

    out_path = args.output
    if out_path is None:
        out_path = state_path.with_name(f"{state_path.stem}_router_config.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the base configuration so permutation evaluation (if any) has access to it.
    out_path.write_text(json.dumps(base_cfg, indent=2), encoding="utf-8")

    base_summary: Optional[Dict[str, Any]] = None
    dynamic_summary: Optional[Dict[str, Any]] = None

    permutation_output = args.permutation_output or out_path.with_suffix(".permutation.json")
    permutation_output = Path(permutation_output)
    metric_key = "p_value_min" if args.pvalue_metric == "min" else "p_value_mean"

    if args.domain_root is not None:
        base_summary = summarise_domain(
            domain_root=args.domain_root,
            config_path=out_path,
            output_path=permutation_output,
            iterations=args.permutation_iterations,
        )

    base_metric_value: Optional[float] = None
    if base_summary is not None:
        metric_raw = base_summary.get(metric_key)
        if isinstance(metric_raw, (int, float)):
            base_metric_value = float(metric_raw)

    base_candidate = {
        "cfg": base_cfg,
        "percentiles": base_percentiles,
        "coverage": base_coverage,
        "summary": base_summary,
        "target": base_coverage,
        "metric_value": base_metric_value,
        "mode": "base",
    }

    optimised_candidate, optimisation_rows = optimise_permutation_guardrail(
        metrics=metrics,
        args=args,
        base_candidate=base_candidate,
        out_path=out_path,
        permutation_output=permutation_output,
    )

    final_cfg = optimised_candidate.get("cfg", base_cfg)
    final_percentiles = optimised_candidate.get("percentiles", base_percentiles)
    final_coverage = float(optimised_candidate.get("coverage", base_coverage))
    final_summary = optimised_candidate.get("summary", base_summary)
    mode = optimised_candidate.get("mode", "base")
    metric_value = optimised_candidate.get("metric_value")
    if metric_value is None and final_summary is not None:
        metric_raw = final_summary.get(metric_key)
        if isinstance(metric_raw, (int, float)):
            metric_value = float(metric_raw)

    if args.domain_root is not None:
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

            previous_suffix = f".{mode}.json"
            previous_config_path = out_path.with_suffix(previous_suffix)
            if not previous_config_path.exists():
                previous_config_path.write_text(json.dumps(final_cfg, indent=2), encoding="utf-8")

            if permutation_output.exists():
                previous_summary_path = permutation_output.with_name(
                    f"{permutation_output.stem}_{mode}{permutation_output.suffix}"
                )
                try:
                    permutation_output.replace(previous_summary_path)
                except OSError:
                    pass

            dynamic_cfg, dynamic_percentiles, dynamic_coverage = compute_configuration(
                metrics,
                dynamic_low,
                dynamic_high,
                extra_coh_qs=args.extra_coherence,
                extra_ent_qs=args.extra_entropy,
                extra_stab_qs=args.extra_stability,
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
            final_summary = dynamic_summary
            prev_metric = metric_float
            if dynamic_summary is not None:
                metric_raw = dynamic_summary.get(metric_key)
                if isinstance(metric_raw, (int, float)):
                    metric_value = float(metric_raw)
                else:
                    metric_value = None
            else:
                metric_value = None
            mode = "dynamic"
            if prev_metric is not None:
                sys.stderr.write(
                    (
                        f"[calibrate_router] Permutation {metric_key}={prev_metric:.3f} > {args.pvalue_threshold:.3f}; "
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
    if final_summary is not None:
        coverage_payload.setdefault("metric", {}).update(
            {
                "key": metric_key,
                "value": metric_value,
                "threshold": args.pvalue_threshold,
            }
        )
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
            }
        )
    if optimisation_rows:
        coverage_payload["optimization"] = {
            "mode": optimised_candidate.get("mode", "base"),
            "candidates": optimisation_rows,
        }
    if args.domain_root is not None:
        coverage_payload["pvalue_metric"] = args.pvalue_metric
        coverage_payload["pvalue_threshold"] = args.pvalue_threshold

    coverage_path.write_text(json.dumps(coverage_payload, indent=2), encoding="utf-8")
    sys.stderr.write(
        f"Calibrated coverage {final_coverage:.4f} using mode '{mode}' -> {out_path}\n"
    )


if __name__ == "__main__":
    main()
