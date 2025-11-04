#!/usr/bin/env python3
"""Fine-grained Logistics sweep to target statistical significance."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


DEFAULT_COVERAGES = [1.5, 1.75, 2.0, 2.25, 2.5]
DEFAULT_ENTROPY = [99.985, 99.987, 99.99]
REPO_ROOT = Path(__file__).resolve().parents[2]


def percentage_to_fraction(value: float) -> float:
    return value / 100.0 if value > 1.0 else value


def format_slug(value: float) -> str:
    # Preserve up to three decimal digits while keeping filenames friendly.
    return f"{value:.3f}".replace(".", "").lstrip("0") or "0"


def run_calibration(
    *,
    state_path: Path,
    domain_root: Path,
    coverage_percent: float,
    entropy_percent: float,
    margin: float,
    iterations: int,
    results_dir: Path,
    use_native: bool,
) -> Dict[str, float]:
    coverage_fraction = percentage_to_fraction(coverage_percent)
    entropy_fraction = percentage_to_fraction(entropy_percent)

    target_low = max(coverage_fraction - margin, 1e-4)
    target_high = coverage_fraction + margin

    coverage_slug = format_slug(coverage_percent)
    entropy_slug = format_slug(entropy_percent)
    config_path = results_dir / f"logistics_cov{coverage_slug}_ent{entropy_slug}_config.json"

    cmd = [
        sys.executable,
        "scripts/calibrate_router.py",
        str(state_path),
        "--target-low",
        f"{target_low:.8f}",
        "--target-high",
        f"{target_high:.8f}",
        "--output",
        str(config_path),
        "--domain-root",
        str(domain_root),
        "--permutation-iterations",
        str(iterations),
        "--optimize-permutation",
        "--extra-entropy",
        f"{entropy_fraction:.8f}",
    ]

    if use_native:
        cmd.append("--use-native-quantum")

    subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    permutation_path = config_path.with_suffix(".permutation.json")
    summary_path = config_path.with_suffix(".coverage.json")

    summary = json.loads(permutation_path.read_text(encoding="utf-8"))["summary"]
    coverage_log = json.loads(summary_path.read_text(encoding="utf-8"))

    return {
        "coverage_target_percent": coverage_percent,
        "entropy_percentile": entropy_percent,
        "coverage_weighted": summary.get("coverage_weighted"),
        "lead_mean": summary.get("lead_mean"),
        "p_value_min": summary.get("p_value_min"),
        "config_path": str(config_path),
        "permutation_path": str(permutation_path),
        "mode": coverage_log.get("mode"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fine-grained Logistics guardrail sweeps")
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("output/planbench_by_domain/logistics/invalid_state.json"),
        help="Path to the Logistics invalid state JSON",
    )
    parser.add_argument(
        "--domain-root",
        type=Path,
        default=Path("output/planbench_by_domain/logistics"),
        help="Domain root containing invalid/metrics for permutation evaluation",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory to store calibration artefacts",
    )
    parser.add_argument(
        "--coverages",
        type=float,
        nargs="*",
        default=DEFAULT_COVERAGES,
        help="Coverage targets (percent)",
    )
    parser.add_argument(
        "--entropy",
        type=float,
        nargs="*",
        default=DEFAULT_ENTROPY,
        help="Entropy percentiles (values in percent)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0005,
        help="Coverage window (fraction) on either side of the target",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20000,
        help="Permutation iterations",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/logistics_sweep_summary.json"),
        help="Path to aggregated summary JSON",
    )
    parser.add_argument(
        "--use-native-quantum",
        action="store_true",
        help="Prefer the native manifold metrics when calibrating",
    )
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, float]] = []
    for coverage in args.coverages:
        for entropy in args.entropy:
            sweep_result = run_calibration(
                state_path=args.state.resolve(),
                domain_root=args.domain_root.resolve(),
                coverage_percent=coverage,
                entropy_percent=entropy,
                margin=args.margin,
                iterations=args.iterations,
                results_dir=args.results_dir.resolve(),
                use_native=args.use_native_quantum,
            )
            results.append(sweep_result)
            print(
                f"coverage={coverage:.3f}% entropy={entropy:.5f}% -> "
                f"p_min={sweep_result['p_value_min']:.5f} lead={sweep_result['lead_mean']}"
            )

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
