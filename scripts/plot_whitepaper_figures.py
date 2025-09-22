#!/usr/bin/env python3
"""Generate the whitepaper figure assets.

Figure 1: Logistics sweep coverage vs permutation p-value.
Figure 2: Live echo count vs hazard λ using warmup manifold snapshots.
Figure 3: STM logistics irreversibility vs λ on enriched domain windows.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_json_array(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list payload in {path}")
    return data


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_figure_synthetic_event_histogram(dataset_path: Path, out_path: Path) -> None:
    records = load_json_array(dataset_path)
    counter: Counter[str] = Counter()
    for row in records:
        analysis = row.get("analysis", {})
        hist = analysis.get("event_histogram", {}) if isinstance(analysis, dict) else {}
        for key, value in hist.items():
            try:
                counter[str(key)] += float(value)
            except (TypeError, ValueError):
                continue

    if not counter:
        raise ValueError(f"Synthetic dataset {dataset_path} does not contain event histograms")

    labels = list(counter.keys())
    counts = [counter[label] for label in labels]

    plt.figure(figsize=(6.0, 3.6))
    plt.bar(labels, counts, color="#6366f1")
    plt.xlabel("QFH event state")
    plt.ylabel("Frequency across synthetic samples")
    plt.title("Synthetic bitstreams: event composition")
    plt.grid(axis="y", linewidth=0.3, alpha=0.3)
    plt.tight_layout()
    ensure_outdir(out_path.parent)
    plt.savefig(out_path, dpi=160)
    plt.close()


def extract_signal_metrics(signals: Sequence[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    coherence: List[float] = []
    entropy: List[float] = []
    hazard: List[float] = []
    for signal in signals:
        if not isinstance(signal, dict):
            continue
        metrics = signal.get("metrics") if isinstance(signal.get("metrics"), dict) else signal
        try:
            coherence.append(float(metrics.get("coherence", 0.0)))
            entropy.append(float(metrics.get("entropy", 0.0)))
            hazard.append(float(metrics.get("lambda_hazard", metrics.get("lambda", metrics.get("rupture", 0.0)))))
        except (TypeError, ValueError):
            continue
    if not coherence:
        raise ValueError("No metrics extracted from signals payload")
    return np.asarray(coherence), np.asarray(entropy), np.asarray(hazard)


def build_metric_distribution_figures(
    signals: Sequence[dict],
    label: str,
    out_dir: Path,
) -> None:
    ensure_outdir(out_dir)
    coherence, entropy, hazard = extract_signal_metrics(signals)

    def _plot(values: np.ndarray, metric_label: str, filename: str) -> None:
        plt.figure(figsize=(6.0, 3.6))
        plt.hist(values, bins=min(len(values) // 20 + 10, 60), color="#0ea5e9", alpha=0.8)
        plt.xlabel(metric_label)
        plt.ylabel("Frequency")
        plt.title(f"{label}: {metric_label} distribution")
        plt.grid(True, linewidth=0.3, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=160)
        plt.close()

    _plot(coherence, "Coherence", f"{label.lower()}_coherence_hist.png")
    _plot(entropy, "Entropy", f"{label.lower()}_entropy_hist.png")
    _plot(hazard, "Hazard $\\lambda$", f"{label.lower()}_lambda_hist.png")


def build_bridge_summary_figure(metrics_path: Path, out_path: Path) -> None:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    pearson_r = payload.get("pearson_r")
    pearson_p = payload.get("pearson_p")
    spearman_rho = payload.get("spearman_rho")
    spearman_p = payload.get("spearman_p")
    hazard_threshold = payload.get("hazard_threshold")

    plt.figure(figsize=(5.0, 3.0))
    plt.axis("off")
    lines = [
        r"Bridge correlation summary",
        fr"Pearson r = {pearson_r:.3f} (p={pearson_p:.3f})" if pearson_r is not None else "Pearson unavailable",
        fr"Spearman ρ = {spearman_rho:.3f} (p={spearman_p:.3f})" if spearman_rho is not None else "Spearman unavailable",
        fr"Median hazard threshold = {hazard_threshold:.4f}" if hazard_threshold is not None else "",
    ]
    lines = [line for line in lines if line]
    plt.text(0.0, 0.5, "\n".join(lines), fontsize=11, va="center", ha="left")
    ensure_outdir(out_path.parent)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def load_signals_from_state(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    signals = data.get("signals")
    if not isinstance(signals, list):
        raise ValueError(f"{path} does not contain a signals array")
    return signals


def load_signals_from_manifold_dir(paths: Sequence[Path]) -> List[dict]:
    signals: List[dict] = []
    for root in paths:
        if root.is_file() and root.suffix == ".json":
            payload = json.loads(root.read_text(encoding="utf-8"))
            sigs = payload.get("signals")
            if isinstance(sigs, list):
                signals.extend(dict(s) for s in sigs if isinstance(s, dict))
            continue
        if root.is_dir():
            for entry in root.rglob("*.json"):
                try:
                    payload = json.loads(entry.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                sigs = payload.get("signals")
                if isinstance(sigs, list):
                    signals.extend(dict(s) for s in sigs if isinstance(s, dict))
    if not signals:
        raise ValueError("No signals found in provided manifold paths")
    return signals


def build_figure_logistics_sweep(
    sweep_path: Path,
    out_path: Path,
    hist_out: Path | None = None,
) -> None:
    rows = load_json_array(sweep_path)
    records: List[Tuple[float, float]] = []
    p_vals: List[float] = []
    for row in rows:
        coverage_target = float(row.get("coverage_target_percent", 0.0))
        p_min = row.get("p_value_min")
        if p_min is None:
            continue
        try:
            p_val = float(p_min)
        except (TypeError, ValueError):
            continue
        records.append((coverage_target, p_val))
        p_vals.append(p_val)

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

    if hist_out is not None and p_vals:
        plt.figure(figsize=(6.0, 3.6))
        plt.hist(p_vals, bins=min(len(p_vals), 20), color="#1a73e8", alpha=0.8)
        plt.axhline(0.05, linestyle="--", color="#d93025", linewidth=1)
        plt.xlabel("Permutation $p_{min}$")
        plt.ylabel("Frequency")
        plt.title("Logistics sweep permutation distribution")
        plt.grid(True, linewidth=0.3, alpha=0.3)
        plt.tight_layout()
        plt.savefig(hist_out, dpi=160)
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


def build_figure_echo_vs_lambda(
    warmup_dir: Path,
    out_path: Path,
    snapshot_out: Path | None = None,
    lambda_curve_out: Path | None = None,
) -> None:
    records: List[Tuple[float, float, float, int]] = []
    for signal in iter_warmup_signals(warmup_dir):
        repetition = signal.get("repetition") or {}
        try:
            count = float(repetition.get("count_1h"))
            hazard = float(signal.get("lambda_hazard") or signal.get("coeffs", {}).get("lambda_hazard") or signal.get("coeffs", {}).get("lambda"))
            rupture = float(signal.get("rupture") or signal.get("metrics", {}).get("rupture", 0.0))
            eligible = int(str(signal.get("eligible", 0)).lower() in {"1", "true", "yes"})
        except (TypeError, ValueError):
            continue
        if count is None or hazard is None:
            continue
        records.append((count, hazard, rupture, eligible))

    if not records:
        raise ValueError("No repetition/hazard data extracted from warmup snapshots.")

    counts, hazards, ruptures, eligible_flags = zip(*records)

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
            fh.write("count_1h,lambda,rupture,eligible\n")
            for count, hazard, rupture, eligible in records:
                fh.write(f"{count},{hazard},{rupture},{eligible}\n")

    if lambda_curve_out is not None and hazards:
        ensure_outdir(lambda_curve_out.parent)
        hazard_vals = np.asarray(hazards)
        eligible_vals = np.asarray(eligible_flags)
        bins = np.linspace(hazard_vals.min(), hazard_vals.max(), min(len(hazard_vals), 20))
        indices = np.digitize(hazard_vals, bins)
        totals = np.bincount(indices, minlength=len(bins) + 1)
        hits = np.bincount(indices, weights=eligible_vals, minlength=len(bins) + 1)
        centers = (bins[:-1] + bins[1:]) / 2
        mask = totals[1:-1] > 0
        if mask.any():
            plt.figure(figsize=(6.0, 3.6))
            plt.plot(centers[mask], (hits[1:-1] / totals[1:-1])[mask], linewidth=2, color="#ea4335")
            plt.xlabel("Hazard $\\lambda$")
            plt.ylabel("Admission rate")
            plt.title("Live gate calibration: admission vs $\\lambda$")
            plt.grid(True, linewidth=0.3, alpha=0.3)
            plt.tight_layout()
            plt.savefig(lambda_curve_out, dpi=160)
            plt.close()


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
    parser.add_argument("--synthetic", type=Path, help="Synthetic QFH dataset JSON (from qfh_synthetic.py)")
    parser.add_argument(
        "--planbench-native",
        type=Path,
        help="PlanBench state JSON with native metrics for distribution plots",
    )
    parser.add_argument(
        "--fx-manifolds",
        type=Path,
        nargs="*",
        help="Manifold JSON files or directories for live FX distributions",
    )
    parser.add_argument(
        "--bridge-metrics",
        type=Path,
        help="Bridge metrics JSON emitted by compute_bridge_metrics.py",
    )
    parser.add_argument("--outdir", type=Path, default=Path("score/docs/figures"), help="Output directory for figures")
    parser.add_argument("--note-dir", type=Path, default=Path("score/docs/note"), help="Directory for auxiliary CSV outputs")
    args = parser.parse_args()

    ensure_outdir(args.outdir)
    ensure_outdir(args.note_dir)

    build_figure_logistics_sweep(
        args.sweep,
        args.outdir / "fig1_logistics_sweep.png",
        hist_out=args.outdir / "fig1b_logistics_perm_distribution.png",
    )
    build_figure_echo_vs_lambda(
        args.warmup_dir,
        args.outdir / "fig2_spt_echo_vs_lambda.png",
        snapshot_out=args.note_dir / "eurusd_warmup_snapshot.csv",
        lambda_curve_out=args.outdir / "fig2b_spt_lambda_calibration.png",
    )
    build_figure_irreversibility_vs_lambda(args.logistics_state, args.outdir / "fig3_logistics_irreversibility_vs_lambda.png")

    if args.synthetic:
        build_figure_synthetic_event_histogram(
            args.synthetic,
            args.outdir / "fig0_synthetic_event_hist.png",
        )

    if args.planbench_native:
        planbench_signals = load_signals_from_state(args.planbench_native)
        build_metric_distribution_figures(planbench_signals, "PlanBench", args.outdir / "planbench_metrics")

    if args.fx_manifolds:
        fx_signals = load_signals_from_manifold_dir(args.fx_manifolds)
        build_metric_distribution_figures(fx_signals, "FX", args.outdir / "fx_metrics")

    if args.bridge_metrics:
        build_bridge_summary_figure(args.bridge_metrics, args.outdir / "fig4_bridge_correlation_summary.png")


if __name__ == "__main__":
    main()
