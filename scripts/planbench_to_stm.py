#!/usr/bin/env python3
"""Convert PlanBench-style PDDL traces into STM manifolds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence

from sep_text_manifold.pipeline import analyse_directory
from stm_adapters.pddl_trace import PDDLTraceAdapter


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


def export_traces(
    label: str,
    trace_paths: Sequence[Path],
    adapter: PDDLTraceAdapter,
    output_dir: Path,
    *,
    window_bytes: int,
    stride: int,
    verbose: bool = False,
) -> Path:
    tokens_dir = output_dir / label / "tokens"
    tokens_dir.mkdir(parents=True, exist_ok=True)
    semantic_dir = output_dir / label / "semantic"
    semantic_dir.mkdir(parents=True, exist_ok=True)

    for path in trace_paths:
        struct_path = adapter.run(path, tokens_dir)
        semantic_source = struct_path.with_name(f"{path.stem}{adapter.semantic_suffix}")
        if semantic_source.exists():
            semantic_source.replace(semantic_dir / semantic_source.name)
    result = analyse_directory(
        str(tokens_dir),
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

    state_path = output_dir / f"{label}_state.json"
    manifold_path = output_dir / f"{label}_manifold.json"
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    manifold_path.write_text(json.dumps(result.signals, indent=2), encoding="utf-8")
    if verbose:
        print(f"Exported {label} traces → {state_path}")
    return state_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PlanBench ➜ STM exporter")
    parser.add_argument("--valid", nargs="+", required=True, help="Valid trace files or directories")
    parser.add_argument("--invalid", nargs="+", help="Invalid trace files or directories")
    parser.add_argument("--output", required=True, help="Output directory for STM corpora")
    parser.add_argument("--window-bytes", type=int, default=512, help="Sliding window size in bytes")
    parser.add_argument("--stride", type=int, default=256, help="Stride length in bytes")
    parser.add_argument("--verbose", action="store_true", help="Print progress information")
    args = parser.parse_args(argv)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = PDDLTraceAdapter()
    valid_paths = _collect_paths(args.valid)
    export_traces(
        "gold",
        valid_paths,
        adapter,
        output_dir,
        window_bytes=args.window_bytes,
        stride=args.stride,
        verbose=args.verbose,
    )
    if args.invalid:
        invalid_paths = _collect_paths(args.invalid)
        export_traces(
            "invalid",
            invalid_paths,
            adapter,
            output_dir,
            window_bytes=args.window_bytes,
            stride=args.stride,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
