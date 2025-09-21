#!/usr/bin/env python3
"""Compute guardrail coverage and permutation p-values using calibrated router configs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def permutation_p_value(
    *,
    window_count: int,
    alerts_count: int,
    failure_index: int,
    observed_lead: int,
    iterations: int = 1000,
    seed: int = 1337,
) -> float:
    if window_count <= 0 or alerts_count <= 0 or observed_lead is None:
        return 1.0
    rng = random.Random(seed)
    hits = 0
    sample_size = min(alerts_count, window_count)
    if sample_size == 0:
        return 1.0
    for _ in range(iterations):
        sample = rng.sample(range(window_count), sample_size)
        sample_leads = [max(0, failure_index - idx) for idx in sample]
        if not sample_leads:
            continue
        sim_lead = min(sample_leads)
        if sim_lead <= observed_lead:
            hits += 1
    return hits / iterations if iterations else 1.0


def compute_alert_indices(
    state_path: Path,
    thresholds: Dict[str, float],
    *,
    limit: Optional[int] = None,
) -> List[int]:
    state = load_json(state_path)
    signals = state.get("signals") or []
    if limit is not None:
        signals = signals[:limit]

    alerts: List[int] = []
    min_coh = float(thresholds.get("min_coh", 0.0))
    max_ent = float(thresholds.get("max_ent", 1.0))
    min_stab = float(thresholds.get("min_stab", 0.0))

    for idx, signal in enumerate(signals):
        if not isinstance(signal, dict):
            continue
        metrics = signal.get("metrics") or {}
        coh = float(metrics.get("coherence", 0.0))
        ent = float(metrics.get("entropy", 1.0))
        stab = float(metrics.get("stability", 0.0))
        if coh >= min_coh and ent <= max_ent and stab >= min_stab:
            alerts.append(idx)
    return alerts


def summarise_domain(
    *,
    domain_root: Path,
    config_path: Path,
    output_path: Path,
    iterations: int,
) -> Dict[str, float]:
    config = load_json(config_path)
    thresholds = config.get("router", {}).get("foreground", {})
    if not thresholds:
        raise ValueError(f"Router thresholds missing in {config_path}")

    metrics_dir = domain_root / "invalid" / "metrics"
    states_dir = domain_root / "invalid" / "states"
    records: List[Dict[str, float]] = []

    for lead_file in sorted(metrics_dir.glob("*.trace_lead.json")):
        lead_data = load_json(lead_file)
        trace_name = lead_data.get("trace")
        failure_index = int(lead_data.get("failure_index", 0))
        window_count = int(lead_data.get("window_count", 0))
        state_path = states_dir / lead_file.name.replace(".trace_lead.json", ".trace_state.json")
        if not state_path.exists():
            continue
        alerts = compute_alert_indices(state_path, thresholds, limit=window_count)
        coverage = len(alerts) / window_count if window_count else 0.0
        leads = [failure_index - idx for idx in alerts if idx <= failure_index]
        observed = min(leads) if leads else None
        pval = permutation_p_value(
            window_count=window_count,
            alerts_count=len(alerts),
            failure_index=failure_index,
            observed_lead=observed or 0,
            iterations=iterations,
        ) if observed is not None else 1.0
        records.append(
            {
                "trace": trace_name,
                "coverage": coverage,
                "lead": observed,
                "p_value": pval,
                "alerts": len(alerts),
                "window_count": window_count,
            }
        )

    total_windows = sum(r["window_count"] for r in records)
    total_alerts = sum(r["alerts"] for r in records)
    coverage_values = [r["coverage"] for r in records]
    lead_values = [r["lead"] for r in records if isinstance(r["lead"], (int, float)) and r["lead"] is not None]
    pval_values = [r["p_value"] for r in records]

    summary = {
        "trace_count": len(records),
        "total_windows": total_windows,
        "total_alerts": total_alerts,
        "coverage_weighted": (total_alerts / total_windows) if total_windows else 0.0,
        "coverage_mean": (sum(coverage_values) / len(coverage_values)) if coverage_values else 0.0,
        "coverage_min": min(coverage_values) if coverage_values else 0.0,
        "coverage_max": max(coverage_values) if coverage_values else 0.0,
        "lead_mean": mean(lead_values) if lead_values else None,
        "lead_min": min(lead_values) if lead_values else None,
        "lead_max": max(lead_values) if lead_values else None,
        "lead_hits": len(lead_values),
        "p_value_mean": mean(pval_values) if pval_values else None,
        "p_value_median": (
            sorted(pval_values)[len(pval_values) // 2] if pval_values else None
        ),
        "p_value_min": min(pval_values) if pval_values else None,
        "p_value_max": max(pval_values) if pval_values else None,
    }

    payload = {
        "config": str(config_path),
        "thresholds": thresholds,
        "records": records,
        "summary": summary,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute guardrail coverage and permutation p-values")
    parser.add_argument("domain_root", type=Path, help="Domain export root (e.g., output/planbench_by_domain/blocksworld)")
    parser.add_argument("config", type=Path, help="Router config JSON produced by calibrate_router.py")
    parser.add_argument("--iterations", type=int, default=1000, help="Permutation iterations")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON report path")
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        output_path = args.domain_root / "permutation_report.json"

    summary = summarise_domain(
        domain_root=args.domain_root,
        config_path=args.config,
        output_path=output_path,
        iterations=args.iterations,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
