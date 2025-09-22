#!/usr/bin/env python3
"""Compute STM↔spt bridge numerics for the whitepaper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.stats import pearsonr, spearmanr


def load_state(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compute_bridge_metrics(state_path: Path, config_path: Path, out_dir: Path) -> None:
    state = load_state(state_path)
    cfg = load_config(config_path)
    thresholds = cfg.get("router", {}).get("foreground", {})
    min_coh = float(thresholds.get("min_coh", 0.0))
    max_ent = float(thresholds.get("max_ent", 1.0))
    min_stab = float(thresholds.get("min_stab", 0.0))

    signals = state.get("signals", [])
    irreversibility: list[float] = []
    hazards: list[float] = []
    logistic_pass: list[int] = []

    for window in signals:
        if not isinstance(window, dict):
            continue
        metrics = window.get("metrics", {})
        features = window.get("features", {}).get("logistics", {})
        irr = features.get("logistics_irreversibility")
        hazard = metrics.get("lambda_hazard", metrics.get("lambda"))
        if irr is None or hazard is None:
            continue
        try:
            irr = float(irr)
            hazard = float(hazard)
        except (TypeError, ValueError):
            continue
        irreversibility.append(irr)
        hazards.append(hazard)

        coh = float(metrics.get("coherence", 0.0))
        ent = float(metrics.get("entropy", 1.0))
        stab = float(metrics.get("stability", 0.0))
        logistic_pass.append(int(coh >= min_coh and ent <= max_ent and stab >= min_stab))

    if not irreversibility:
        raise ValueError("No irreversibility/hazard pairs found in state file")

    irr_arr = np.asarray(irreversibility)
    haz_arr = np.asarray(hazards)
    pass_arr = np.asarray(logistic_pass)

    pearson = pearsonr(irr_arr, haz_arr)
    spearman = spearmanr(irr_arr, haz_arr)

    hazard_threshold = float(np.percentile(haz_arr, 50))  # median cutoff
    hazard_low = haz_arr <= hazard_threshold
    contingency = {
        "logistic_pass": int(pass_arr[hazard_low].sum()),
        "logistic_fail": int((1 - pass_arr)[hazard_low].sum()),
        "hazard_threshold": hazard_threshold,
        "total_entries": int(len(haz_arr)),
    }
    table = {
        "hazard_threshold": hazard_threshold,
        "pass_low_hazard": int((pass_arr & hazard_low).sum()),
        "fail_low_hazard": int(((1 - pass_arr) & hazard_low).sum()),
        "pass_high_hazard": int((pass_arr & (~hazard_low)).sum()),
        "fail_high_hazard": int(((1 - pass_arr) & (~hazard_low)).sum()),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bridge_metrics.json").write_text(
        json.dumps(
            {
                "pearson_r": pearson.statistic,
                "pearson_p": pearson.pvalue,
                "spearman_rho": spearman.statistic,
                "spearman_p": spearman.pvalue,
                "hazard_threshold": hazard_threshold,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "bridge_contingency.json").write_text(json.dumps(table, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute bridge numerics for whitepaper")
    parser.add_argument("--state", type=Path, required=True, help="Enriched logistics state JSON")
    parser.add_argument("--config", type=Path, required=True, help="Router config JSON")
    parser.add_argument("--output", type=Path, default=Path("score/docs/note"), help="Output directory")
    args = parser.parse_args()

    compute_bridge_metrics(args.state, args.config, args.output)


if __name__ == "__main__":
    main()
