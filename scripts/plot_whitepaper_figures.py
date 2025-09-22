#!/usr/bin/env python3
"""Generate the whitepaper figure assets.

Figure 1: Logistics sweep coverage vs permutation p-value.
Figure 2: Live echo count vs hazard λ using warmup manifold snapshots.
Figure 3: STM logistics irreversibility vs λ on enriched domain windows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_json_array(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list payload in {path}")
    return data


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_figure_logistics_sweep(sweep_path: Path, out_path: Path) -> None:
    rows = load_json_array(sweep_path)
    records: List[Tuple[float, float]] = []
    for row in rows:
        coverage_target = float(row.get("coverage_target_percent", 0.0))
        p_min = row.get("p_value_min")
        if p_min is None:
            continue
        try:
            p_min = float(p_min)
        except (TypeError, ValueError):
            continue
        records.append((coverage_target, p_min))

    if not records:
        raise ValueError("No sweep records with p-values were found.")

    records.sort()
    coverage, p_values = zip(*records)

    plt.figure(figsize=(6.0, 3.6))
    plt.plot(coverage, p_values, marker="o", color="#1a73e8", linewidth=2)
    plt.axhline(0.05, linestyle="--", color="#d93025", linewidth=1)
    plt.xlabel("Coverage target (percent)")
    plt.ylabel("Permutation $p_{min}$")
    plt.title("Logistics sweep (1.6–2.2% coverage)")
    plt.grid(True, linewidth=0.3, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def iter_warmup_signals(warmup_dir: Path) -> Iterable[dict]:
    if not warmup_dir.exists():
        raise FileNotFoundError(f"Warmup directory not found: {warmup_dir}")
    for path in sorted(warmup_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        signals = payload.get("signals")
        if not isinstance(signals, list):
            continue
        for signal in signals:
            if isinstance(signal, dict):
                yield signal


def build_figure_echo_vs_lambda(warmup_dir: Path, out_path: Path, snapshot_out: Path | None = None) -> None:
    records: List[Tuple[float, float, float]] = []
    for signal in iter_warmup_signals(warmup_dir):
        repetition = signal.get("repetition") or {}
        try:
            count = float(repetition.get("count_1h"))
            hazard = float(signal.get("lambda_hazard") or signal.get("coeffs", {}).get("lambda_hazard") or signal.get("coeffs", {}).get("lambda"))
            rupture = float(signal.get("rupture") or signal.get("metrics", {}).get("rupture", 0.0))
        except (TypeError, ValueError):
            continue
        if count is None or hazard is None:
            continue
        records.append((count, hazard, rupture))

    if not records:
        raise ValueError("No repetition/hazard data extracted from warmup snapshots.")

    counts, hazards, ruptures = zip(*records)

    plt.figure(figsize=(6.0, 3.6))
    plt.scatter(counts, hazards, s=10, alpha=0.6, color="#34a853")
    plt.xlabel("Echo count (last 1h)")
    plt.ylabel("Hazard $\\lambda$")
    plt.title("EUR/USD manifolds: echo vs hazard")
    plt.grid(True, linewidth=0.3, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

    if snapshot_out is not None:
        ensure_outdir(snapshot_out.parent)
        with snapshot_out.open("w", encoding="utf-8") as fh:
            fh.write("count_1h,lambda,rupture\n")
            for count, hazard, rupture in records:
                fh.write(f"{count},{hazard},{rupture}\n")


def build_figure_irreversibility_vs_lambda(state_path: Path, out_path: Path) -> None:
    data = json.loads(state_path.read_text(encoding="utf-8"))
    signals = data.get("signals")
    if not isinstance(signals, list):
        raise ValueError(f"{state_path} does not contain a signals array")

    rows: List[Tuple[float, float]] = []
    for window in signals:
        if not isinstance(window, dict):
            continue
        metrics = window.get("metrics") or {}
        features = window.get("features", {}).get("logistics", {})
        try:
            irr = float(features.get("logistics_irreversibility"))
            hazard = float(metrics.get("lambda_hazard", metrics.get("lambda", 0.0)))
        except (TypeError, ValueError):
            continue
        rows.append((irr, hazard))

    if not rows:
        raise ValueError("No logistics irreversibility/λ pairs found in state file")

    irr, hazard = zip(*rows)
    plt.figure(figsize=(6.0, 3.6))
    plt.scatter(irr, hazard, s=10, alpha=0.6, color="#f9ab00")
    plt.xlabel("STM irreversibility")
    plt.ylabel("Hazard $\\lambda$")
    plt.title("Logistics windows: irreversibility vs λ")
    plt.grid(True, linewidth=0.3, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate whitepaper figures")
    parser.add_argument("--sweep", type=Path, required=True, help="Sweep summary JSON for Logistics")
    parser.add_argument("--warmup-dir", type=Path, required=True, help="Directory containing warmup manifolds")
    parser.add_argument("--logistics-state", type=Path, required=True, help="Enriched logistics invalid state JSON")
    parser.add_argument("--outdir", type=Path, default=Path("score/docs/figures"), help="Output directory for figures")
    parser.add_argument("--note-dir", type=Path, default=Path("score/docs/note"), help="Directory for auxiliary CSV outputs")
    args = parser.parse_args()

    ensure_outdir(args.outdir)
    ensure_outdir(args.note_dir)

    build_figure_logistics_sweep(args.sweep, args.outdir / "fig1_logistics_sweep.png")
    build_figure_echo_vs_lambda(
        args.warmup_dir,
        args.outdir / "fig2_spt_echo_vs_lambda.png",
        snapshot_out=args.note_dir / "eurusd_warmup_snapshot.csv",
    )
    build_figure_irreversibility_vs_lambda(args.logistics_state, args.outdir / "fig3_logistics_irreversibility_vs_lambda.png")


if __name__ == "__main__":
    main()
