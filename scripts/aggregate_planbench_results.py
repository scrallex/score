#!/usr/bin/env python3
"""Aggregate STM PlanBench experiment outputs into a domain-level scorecard."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_INPUT_ROOT = Path("output/planbench_public")
DEFAULT_OUTPUT = Path("docs/note/planbench_scorecard.csv")
DECISIVE_PATH = 0.3
DECISIVE_SIGNAL = 0.4


def infer_domain(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith("bw_") or "blocksworld" in lowered:
        return "Blocksworld"
    if lowered.startswith("mystery"):
        return "Mystery Blocksworld"
    if lowered.startswith("logistics"):
        return "Logistics"
    return name


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_trace_files(trace_root: Path, domains: Sequence[str], traces_dir: str) -> Iterable[Tuple[str, Path, Dict[str, object]]]:
    for domain in domains:
        traces_path = trace_root / domain / traces_dir
        if not traces_path.exists():
            continue
        for trace_path in sorted(traces_path.glob("*.json")):
            data = load_json(trace_path)
            yield domain, trace_path, data


def decisive_percentage(state_path: Path) -> Optional[float]:
    if not state_path.exists():
        return None
    state = load_json(state_path)
    signals = state.get("signals")
    if not isinstance(signals, list) or not signals:
        return None
    decisive = 0
    total = 0
    for sig in signals:
        if not isinstance(sig, dict):
            continue
        dilution = sig.get("dilution", {})
        if not isinstance(dilution, dict):
            continue
        path_val = float(dilution.get("path", 0.0))
        signal_val = float(dilution.get("signal", 0.0))
        total += 1
        if path_val < DECISIVE_PATH and signal_val < DECISIVE_SIGNAL:
            decisive += 1
    if total == 0:
        return None
    return decisive / total


def bootstrap_ci(values: Sequence[float], *, iterations: int = 1000, seed: int = 1234) -> Tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    mean_val = float(statistics.mean(values))
    if len(values) == 1:
        return mean_val, mean_val, mean_val
    rng = random.Random(seed)
    samples: List[float] = []
    for _ in range(iterations):
        resample = [rng.choice(values) for _ in values]
        samples.append(float(statistics.mean(resample)))
    samples.sort()
    lo_idx = int(0.025 * iterations)
    hi_idx = int(0.975 * iterations)
    return mean_val, samples[lo_idx], samples[min(hi_idx, iterations - 1)]


def aggregate(input_root: Path, *, trace_root: Optional[Path], traces_dir: str) -> List[Dict[str, object]]:
    metrics_summary_path = input_root / "invalid" / "metrics" / "summary.json"
    if not metrics_summary_path.exists():
        raise FileNotFoundError(f"Metrics summary not found at {metrics_summary_path}")
    metrics_data = load_json(metrics_summary_path)
    lead_records = {rec.get("trace"): rec for rec in metrics_data.get("lead_records", [])}
    twin_records = {rec.get("trace"): rec for rec in metrics_data.get("twin_records", [])}
    path_threshold = float(metrics_data.get("path_threshold", 0.0))
    signal_threshold = float(metrics_data.get("signal_threshold", 0.0))

    run_summary_path = input_root / "run_summary.json"
    if trace_root is None and run_summary_path.exists():
        run_summary = load_json(run_summary_path)
        trace_root_str = run_summary.get("trace_root")
        if trace_root_str:
            trace_root = Path(trace_root_str)
    if trace_root is None:
        raise ValueError("Trace root could not be inferred; pass --trace-root or ensure run_summary.json contains it")

    domains = [d.name for d in sorted(trace_root.iterdir()) if d.is_dir()]
    domain_rows: Dict[str, Dict[str, object]] = {}
    domain_valid_counts: Dict[str, List[bool]] = {domain: [] for domain in domains}
    domain_corrupt_traces: Dict[str, List[str]] = {domain: [] for domain in domains}
    domain_trace_paths: Dict[str, Dict[str, Path]] = {domain: {} for domain in domains}

    for domain, trace_path, trace_data in iter_trace_files(trace_root, domains, traces_dir):
        status = str(trace_data.get("status", "")).lower()
        plan_type = trace_data.get("plan_type") or ("corrupt" if status == "invalid" else "valid")
        trace_name = trace_path.stem
        domain_trace_paths[domain][trace_name] = trace_path
        if plan_type == "valid":
            domain_valid_counts[domain].append(status == "valid")
        else:
            domain_corrupt_traces[domain].append(trace_name)

    results: List[Dict[str, object]] = []
    for domain in domains:
        valid_flags = domain_valid_counts.get(domain, [])
        n_valid = len(valid_flags)
        plan_accuracy = (sum(valid_flags) / n_valid) if n_valid else 0.0

        corrupt_traces = domain_corrupt_traces.get(domain, [])
        lead_means: List[float] = []
        lead_coverages: List[float] = []
        lead_max_vals: List[float] = []
        lead_min_vals: List[float] = []
        twin_min_distances: List[float] = []
        twin_counts = {0.3: 0, 0.4: 0, 0.5: 0}
        decisive_values: List[float] = []

        for trace_name in corrupt_traces:
            lead_record = lead_records.get(trace_name)
            if lead_record:
                stats = lead_record.get("stats", {})
                lead_means.append(float(stats.get("mean", 0.0)))
                lead_coverages.append(float(stats.get("coverage", 0.0)))
                leads = stats.get("leads", []) or []
                if leads:
                    lead_max_vals.append(max(leads))
                    lead_min_vals.append(min(leads))
            twin_record = twin_records.get(trace_name)
            if twin_record:
                suggestions = []
                for detail in twin_record.get("details", []):
                    for suggestion in detail.get("suggestions", []):
                        distance = float(suggestion.get("distance", 1.0))
                        suggestions.append(distance)
                if suggestions:
                    best = min(suggestions)
                    twin_min_distances.append(best)
                    for tau in twin_counts:
                        if best <= tau:
                            twin_counts[tau] += 1
            # decisive percentage
            state_path = input_root / "invalid" / "states" / f"{trace_name}.json"
            decisive = decisive_percentage(state_path)
            if decisive is not None:
                decisive_values.append(decisive)

        # Aggregated metrics
        lead_mean = float(statistics.mean(lead_means)) if lead_means else 0.0
        lead_coverage = float(statistics.mean(lead_coverages)) if lead_coverages else 0.0
        lead_max = max(lead_max_vals) if lead_max_vals else 0.0
        lead_min = min(lead_min_vals) if lead_min_vals else 0.0
        twin_rates = {tau: (twin_counts[tau] / len(corrupt_traces) if corrupt_traces else 0.0) for tau in twin_counts}
        decisive_pct = float(statistics.mean(decisive_values)) if decisive_values else 0.0
        ann_mean, ann_lo, ann_hi = bootstrap_ci(twin_min_distances)

        results.append(
            {
                "domain": infer_domain(domain),
                "n_traces": n_valid,
                "plan_accuracy": plan_accuracy,
                "lead_mean": lead_mean,
                "lead_coverage": lead_coverage,
                "lead_max": lead_max,
                "lead_min": lead_min,
                "twin_rate@0.3": twin_rates[0.3],
                "twin_rate@0.4": twin_rates[0.4],
                "twin_rate@0.5": twin_rates[0.5],
                "decisive_pct": decisive_pct,
                "ann_mean": ann_mean,
                "ann_ci95_lo": ann_lo,
                "ann_ci95_hi": ann_hi,
                "path_threshold": path_threshold,
                "signal_threshold": signal_threshold,
            }
        )

    results.sort(key=lambda item: item["domain"])
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate STM PlanBench metrics")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--trace-root", type=Path, help="Override trace root (defaults to run_summary entry)")
    parser.add_argument("--traces-dir", default="traces")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = aggregate(args.input_root, trace_root=args.trace_root, traces_dir=args.traces_dir)
    if not rows:
        raise SystemExit("No results to aggregate")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
