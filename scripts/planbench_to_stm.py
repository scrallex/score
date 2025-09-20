#!/usr/bin/env python3
"""Convert PlanBench-style traces into STM manifolds and analyse alert lead time."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

try:  # Plotting is optional but recommended
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:  # pragma: no cover - plotting optional
    plt = None

from sep_text_manifold.pipeline import analyse_directory
from sep_text_manifold.comparison import detection_lead_time
from sep_text_manifold.feedback import suggest_twin_action
from stm_adapters.pddl_trace import PDDLTraceAdapter


@dataclass
class TraceContext:
    name: str
    trace_path: Path
    state_path: Path
    summary_path: Path
    summary: Dict[str, Any]
    signals: List[Dict[str, Any]]
    transitions: List[Mapping[str, Any]]


def _load_transitions(path: Path) -> List[Mapping[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    transitions = data.get("transitions") or data.get("trace") or []
    if not isinstance(transitions, Sequence):
        raise ValueError(f"Trace file {path} does not contain a transitions list")
    return [t for t in transitions if isinstance(t, Mapping)]


def _transition_is_valid(transition: Mapping[str, Any]) -> bool:
    if "valid" in transition:
        return bool(transition["valid"])
    if "is_valid" in transition:
        return bool(transition["is_valid"])
    status = transition.get("status")
    if isinstance(status, str):
        status_lower = status.lower()
        if status_lower in {"valid", "ok", "success", "accepted"}:
            return True
        if status_lower in {"invalid", "fail", "failure", "error"}:
            return False
    if transition.get("errors") or transition.get("error"):
        return False
    return True


def _invalid_indices(transitions: Sequence[Mapping[str, Any]]) -> List[int]:
    return [idx for idx, transition in enumerate(transitions) if not _transition_is_valid(transition)]


def _alert_indices(signals: Sequence[Mapping[str, Any]], *, path_threshold: float, signal_threshold: float, limit: int) -> List[int]:
    alerts: List[int] = []
    for idx in range(limit):
        dilution = signals[idx].get("dilution", {}) if isinstance(signals[idx], Mapping) else {}
        path_value = float(dilution.get("path", 0.0))
        signal_value = float(dilution.get("signal", 0.0))
        if path_value >= path_threshold or signal_value >= signal_threshold:
            alerts.append(idx)
    return alerts


def _plot_dilution(
    *,
    ctx: TraceContext,
    alerts: Sequence[int],
    failures: Sequence[int],
    plot_path: Path,
    path_threshold: float,
    signal_threshold: float,
    limit: int,
) -> None:
    if plt is None:
        return
    path_series = [float((ctx.signals[idx].get("dilution", {}) or {}).get("path", 0.0)) for idx in range(limit)]
    signal_series = [float((ctx.signals[idx].get("dilution", {}) or {}).get("signal", 0.0)) for idx in range(limit)]
    steps = list(range(limit))

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(steps, path_series, label="Path Dilution", color="#3b82f6")
    plt.plot(steps, signal_series, label="Signal Dilution", color="#22c55e")
    plt.axhline(path_threshold, color="#1d4ed8", linestyle="--", linewidth=1, alpha=0.6, label="Path threshold")
    plt.axhline(signal_threshold, color="#15803d", linestyle="--", linewidth=1, alpha=0.6, label="Signal threshold")

    for idx in alerts:
        plt.axvline(idx, color="#f97316", alpha=0.2, linewidth=1)
    for idx in failures:
        plt.axvline(idx, color="#ef4444", linestyle="--", alpha=0.4, linewidth=1.5)

    plt.title(f"Dilution ramps – {ctx.name}")
    plt.xlabel("Step index")
    plt.ylabel("Dilution")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


def _compute_lead_metrics(
    *,
    ctx: TraceContext,
    path_threshold: float,
    signal_threshold: float,
    metrics_dir: Path,
) -> Optional[Dict[str, Any]]:
    limit = min(len(ctx.signals), len(ctx.transitions))
    if limit == 0:
        return None
    alerts = _alert_indices(ctx.signals, path_threshold=path_threshold, signal_threshold=signal_threshold, limit=limit)
    failures = [idx for idx in _invalid_indices(ctx.transitions) if idx < limit]
    stats = detection_lead_time(alerts, failures)

    record: Dict[str, Any] = {
        "trace": ctx.name,
        "alerts": alerts,
        "val_failures": failures,
        "stats": asdict(stats),
        "window_count": limit,
    }
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{ctx.name}_lead.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    plot_path = metrics_dir / "plots" / f"{ctx.name}_dilution.png"
    _plot_dilution(
        ctx=ctx,
        alerts=alerts,
        failures=failures,
        plot_path=plot_path,
        path_threshold=path_threshold,
        signal_threshold=signal_threshold,
        limit=limit,
    )
    record["plot"] = str(plot_path.relative_to(metrics_dir)) if plot_path.exists() else None
    return record


def _compute_twin_metrics(
    *,
    ctx: TraceContext,
    gold_state: Mapping[str, Any],
    twin_distance: float,
    top_k: int,
    match_signature: bool,
    metrics_dir: Path,
) -> Dict[str, Any]:
    limit = min(len(ctx.signals), len(ctx.transitions))
    failures = [idx for idx in _invalid_indices(ctx.transitions) if idx < limit]
    successes = 0
    details: List[Dict[str, Any]] = []
    for idx in failures:
        invalid_signal = ctx.signals[idx]
        suggestions = suggest_twin_action(
            invalid_signal,
            gold_state,
            top_k=top_k,
            max_distance=twin_distance,
            match_signature=match_signature,
        )
        serialised = [
            {
                "window_id": suggestion.window_id,
                "signature": suggestion.signature,
                "distance": suggestion.distance,
                "tokens": list(suggestion.tokens),
                "metrics": suggestion.metrics,
            }
            for suggestion in suggestions
        ]
        details.append({"index": idx, "suggestions": serialised})
        if any(suggestion.distance <= twin_distance for suggestion in suggestions):
            successes += 1

    total = len(failures)
    record = {
        "trace": ctx.name,
        "total_failures": total,
        "corrected": successes,
        "correction_rate": successes / total if total else 0.0,
        "details": details,
    }
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{ctx.name}_twin.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def _aggregate_lead(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {}
    coverage = [rec["stats"].get("coverage", 0.0) for rec in records]
    all_leads: List[int] = []
    for rec in records:
        all_leads.extend(rec["stats"].get("leads", []) or [])
    summary: Dict[str, Any] = {
        "count": len(records),
        "coverage_mean": sum(coverage) / len(coverage) if coverage else 0.0,
        "coverage_min": min(coverage) if coverage else 0.0,
        "coverage_max": max(coverage) if coverage else 0.0,
        "alerts_mean": sum(len(rec["alerts"]) for rec in records) / len(records),
        "failures_mean": sum(len(rec["val_failures"]) for rec in records) / len(records),
    }
    if all_leads:
        summary.update(
            {
                "lead_mean": sum(all_leads) / len(all_leads),
                "lead_min": min(all_leads),
                "lead_max": max(all_leads),
                "lead_count": len(all_leads),
            }
        )
    return summary


def _aggregate_twin(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {}
    total_failures = sum(rec["total_failures"] for rec in records)
    total_corrected = sum(rec["corrected"] for rec in records)
    return {
        "count": len(records),
        "total_failures": total_failures,
        "total_corrected": total_corrected,
        "overall_correction_rate": total_corrected / total_failures if total_failures else 0.0,
        "mean_rate": sum(rec["correction_rate"] for rec in records) / len(records),
    }


def export_traces(
    label: str,
    trace_paths: Sequence[Path],
    adapter: PDDLTraceAdapter,
    output_root: Path,
    *,
    window_bytes: int,
    stride: int,
    verbose: bool,
) -> tuple[Path, List[TraceContext]]:
    label_root = output_root / label
    tokens_root = label_root / "tokens"
    states_root = label_root / "states"
    tokens_root.mkdir(parents=True, exist_ok=True)
    states_root.mkdir(parents=True, exist_ok=True)

    contexts: List[TraceContext] = []
    for trace_path in trace_paths:
        trace_dir = tokens_root / trace_path.stem
        struct_path = adapter.run(trace_path, trace_dir)
        transitions = _load_transitions(trace_path)
        result = analyse_directory(
            str(trace_dir),
            window_bytes=window_bytes,
            stride=stride,
            extensions=["txt"],
            verbose=verbose,
            min_token_length=1,
            min_alpha_ratio=0.0,
            drop_numeric=False,
            min_occurrences=1,
            cap_tokens_per_window=80,
            graph_min_pmi=0.0,
            graph_max_degree=None,
            theme_min_size=1,
        )

        state = result.to_state(include_signals=True, include_profiles=False)
        summary = result.summary()
        state["summary"] = summary
        state["dilution_summary"] = result.dilution_summary

        state_path = states_root / f"{trace_path.stem}_state.json"
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        summary_path = states_root / f"{trace_path.stem}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        contexts.append(
            TraceContext(
                name=trace_path.stem,
                trace_path=trace_path,
                state_path=state_path,
                summary_path=summary_path,
                summary=summary,
                signals=[dict(sig) for sig in result.signals],
                transitions=transitions,
            )
        )

    aggregated_result = analyse_directory(
        str(tokens_root),
        window_bytes=window_bytes,
        stride=stride,
        extensions=["txt"],
        verbose=verbose,
        min_token_length=1,
        min_alpha_ratio=0.0,
        drop_numeric=False,
        min_occurrences=1,
        cap_tokens_per_window=80,
        graph_min_pmi=0.0,
        graph_max_degree=None,
        theme_min_size=1,
    )
    aggregated_state = aggregated_result.to_state(include_signals=True, include_profiles=False)
    aggregated_state["summary"] = aggregated_result.summary()
    aggregated_state["dilution_summary"] = aggregated_result.dilution_summary
    state_path = output_root / f"{label}_state.json"
    state_path.write_text(json.dumps(aggregated_state, indent=2), encoding="utf-8")
    return state_path, contexts


def process_invalid_traces(
    *,
    contexts: Sequence[TraceContext],
    gold_state_path: Path,
    label_root: Path,
    path_threshold: float,
    signal_threshold: float,
    twin_distance: float,
    twin_top_k: int,
    match_signature: bool,
    verbose: bool,
) -> None:
    if not contexts:
        return
    gold_state = json.loads(gold_state_path.read_text(encoding="utf-8"))
    metrics_root = label_root / "metrics"
    lead_records: List[Dict[str, Any]] = []
    twin_records: List[Dict[str, Any]] = []

    for ctx in contexts:
        if verbose:
            print(f"[stm] analysing invalid trace {ctx.name}")
        lead_record = _compute_lead_metrics(
            ctx=ctx,
            path_threshold=path_threshold,
            signal_threshold=signal_threshold,
            metrics_dir=metrics_root,
        )
        if lead_record is not None:
            lead_records.append(lead_record)
        twin_record = _compute_twin_metrics(
            ctx=ctx,
            gold_state=gold_state,
            twin_distance=twin_distance,
            top_k=twin_top_k,
            match_signature=match_signature,
            metrics_dir=metrics_root,
        )
        twin_records.append(twin_record)

    summary = {
        "path_threshold": path_threshold,
        "signal_threshold": signal_threshold,
        "twin_distance": twin_distance,
        "twin_top_k": twin_top_k,
        "match_signature": match_signature,
        "lead_summary": _aggregate_lead(lead_records),
        "twin_summary": _aggregate_twin(twin_records),
        "lead_records": lead_records,
        "twin_records": twin_records,
    }
    metrics_root.mkdir(parents=True, exist_ok=True)
    (metrics_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _collect_paths(inputs: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.json")))
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(f"Trace input '{item}' not found")
    if not paths:
        raise ValueError("No trace files discovered for the given inputs")
    return paths


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="PlanBench ➜ STM exporter with lead-time analysis")
    parser.add_argument("--valid", nargs="+", required=True, help="Valid trace files or directories")
    parser.add_argument("--invalid", nargs="+", help="Invalid trace files or directories")
    parser.add_argument("--output", required=True, help="Output directory for STM artefacts")
    parser.add_argument("--window-bytes", type=int, default=512, help="Sliding window size in bytes")
    parser.add_argument("--stride", type=int, default=256, help="Stride length in bytes")
    parser.add_argument("--path-threshold", type=float, default=0.55, help="Path dilution threshold for alerts")
    parser.add_argument("--signal-threshold", type=float, default=0.60, help="Signal dilution threshold for alerts")
    parser.add_argument("--twin-distance", type=float, default=0.40, help="Maximum distance for twin corrections")
    parser.add_argument("--twin-top-k", type=int, default=3, help="Number of twin suggestions to retrieve")
    parser.add_argument("--match-signature", action="store_true", help="Only consider twins that share the signature")
    parser.add_argument("--verbose", action="store_true", help="Print progress information")
    args = parser.parse_args(argv)

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    adapter = PDDLTraceAdapter()
    valid_paths = _collect_paths(args.valid)
    gold_state_path, _ = export_traces(
        "gold",
        valid_paths,
        adapter,
        output_root,
        window_bytes=args.window_bytes,
        stride=args.stride,
        verbose=args.verbose,
    )

    if args.invalid:
        invalid_paths = _collect_paths(args.invalid)
        invalid_state_path, invalid_contexts = export_traces(
            "invalid",
            invalid_paths,
            adapter,
            output_root,
            window_bytes=args.window_bytes,
            stride=args.stride,
            verbose=args.verbose,
        )
        process_invalid_traces(
            contexts=invalid_contexts,
            gold_state_path=gold_state_path,
            label_root=output_root / "invalid",
            path_threshold=args.path_threshold,
            signal_threshold=args.signal_threshold,
            twin_distance=args.twin_distance,
            twin_top_k=args.twin_top_k,
            match_signature=args.match_signature,
            verbose=args.verbose,
        )
    else:
        invalid_state_path = None

    run_summary = {
        "gold_state": str(gold_state_path),
        "invalid_state": str(invalid_state_path) if invalid_state_path else None,
        "settings": {
            "window_bytes": args.window_bytes,
            "stride": args.stride,
            "path_threshold": args.path_threshold,
            "signal_threshold": args.signal_threshold,
            "twin_distance": args.twin_distance,
            "twin_top_k": args.twin_top_k,
            "match_signature": args.match_signature,
        },
    }
    (output_root / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
