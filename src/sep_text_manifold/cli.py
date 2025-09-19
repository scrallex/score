"""
Command line interface for the Sep Text Manifold project.

This module uses ``argparse`` to expose a small set of commands for
ingesting a corpus, analysing it and exploring the results.  The CLI
flows through the entire pipeline: reading files, building the
manifold, extracting strings, aggregating metrics, computing scores
and detecting themes.

Results are stored in a JSON file (`stm_state.json` by default) so
that subsequent commands can operate on cached analysis rather than
recomputing everything.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .pipeline import analyse_directory, compute_summary


def _load_state(state_file: Path) -> Dict[str, Any]:
    if state_file.exists():
        with state_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_state(state_file: Path, data: Dict[str, Any]) -> None:
    with state_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _print_summary(summary: Dict[str, Any]) -> None:
    if not summary:
        print("No summary available.")
        return
    file_count = summary.get("file_count")
    token_count = summary.get("token_count")
    corpus_size = summary.get("corpus_size_bytes")
    window_count = summary.get("window_count")
    print("Summary")
    print("-------")
    if file_count is not None:
        print(f"Files analysed: {file_count}")
    if token_count is not None:
        print(f"Tokens extracted: {token_count}")
    if corpus_size is not None:
        print(f"Corpus size (bytes): {corpus_size}")
    if window_count is not None:
        print(f"Manifold windows: {window_count}")
    print(f"Unique strings scored: {summary.get('string_count', 0)}")
    top_patterns = summary.get("top_patternable_strings", [])
    if top_patterns:
        print("\nTop patternable strings:")
        for entry in top_patterns:
            label = entry.get("string", "<unknown>")
            print(
                f"  {label:<24} pattern={entry.get('patternability', 0.0):.3f}"
                f" occ={entry.get('occurrences', 0)}"
            )
    top_connectors = summary.get("top_connectors", [])
    if top_connectors:
        print("\nTop connectors:")
        for entry in top_connectors:
            label = entry.get("string", "<unknown>")
            print(
                f"  {label:<24} connector={entry.get('connector', 0.0):.3f}"
                f" occ={entry.get('occurrences', 0)}"
            )
    if summary.get("theme_count") is not None:
        print(f"\nTheme count: {summary['theme_count']}")
    mean_metrics = summary.get("mean_window_metrics")
    if mean_metrics:
        print("\nMean window metrics:")
        for name, value in mean_metrics.items():
            print(f"  {name}: {value:.3f}")


def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest a directory of text files and analyse it."""
    directory = args.directory
    window_bytes = args.window_bytes
    stride = args.stride
    state_file = Path(args.output)
    result = analyse_directory(
        directory,
        window_bytes=window_bytes,
        stride=stride,
        extensions=args.extensions,
    )
    summary = result.summary(top=args.summary_top)
    state = result.to_state(
        include_signals=args.store_signals,
        include_occurrences=args.store_occurrences,
        include_profiles=args.store_profiles,
    )
    state["summary"] = summary
    state["generated_at"] = datetime.utcnow().isoformat() + "Z"
    _save_state(state_file, state)
    print(f"Analysis complete.  Saved state to {state_file}")
    print()
    _print_summary(summary)


def cmd_strings(args: argparse.Namespace) -> None:
    """List strings sorted by patternability."""
    state = _load_state(Path(args.input))
    scores: Dict[str, Dict[str, Any]] = state.get("string_scores", {})
    if not scores:
        print("No string scores found.  Run 'stm ingest' first.")
        return
    if args.sort == "connector":
        key_func = lambda kv: kv[1].get("connector", 0.0)
    else:
        key_func = lambda kv: kv[1].get("patternability", 0.0)
    sorted_items = sorted(scores.items(), key=key_func, reverse=True)
    top = args.top or len(sorted_items)
    for i, (s, metrics) in enumerate(sorted_items[:top], start=1):
        p_score = metrics.get("patternability", 0.0)
        connector = metrics.get("connector", 0.0)
        occurrences = metrics.get("occurrences", 0)
        print(
            f"{i:3d}. {s:<24} pattern={p_score:.3f}"
            f" connector={connector:.3f} occ={occurrences}"
        )


def cmd_themes(args: argparse.Namespace) -> None:
    """List themes and their member strings."""
    state = _load_state(Path(args.input))
    themes: List[List[str]] = state.get("themes", [])
    if not themes:
        print("No themes found.  Run 'stm ingest' first.")
        return
    for i, members in enumerate(themes, start=1):
        print(f"Theme {i} ({len(members)} strings): {', '.join(sorted(members)[:20])}{'...' if len(members) > 20 else ''}")


def cmd_summary(args: argparse.Namespace) -> None:
    """Display a summary of the latest analysis."""
    state = _load_state(Path(args.input))
    if not state:
        print("No analysis state found.  Run 'stm ingest' first.")
        return
    top_n = args.top or 10
    summary = compute_summary(
        state.get("string_scores", {}),
        signals=state.get("signals"),
        themes=state.get("themes"),
        corpus_size_bytes=state.get("corpus_size_bytes"),
        token_count=state.get("token_count"),
        file_count=len(state.get("files", [])) or state.get("summary", {}).get("file_count"),
        top=top_n,
    )
    _print_summary(summary)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="stm", description="Sep Text Manifold CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    # Ingest command
    p_ingest = subparsers.add_parser("ingest", help="Ingest and analyse a directory of text files")
    p_ingest.add_argument("directory", help="Directory containing text files to analyse")
    p_ingest.add_argument("--window-bytes", dest="window_bytes", type=int, default=2048, help="Size of sliding window in bytes")
    p_ingest.add_argument("--stride", dest="stride", type=int, default=1024, help="Stride between windows in bytes")
    p_ingest.add_argument("--extensions", nargs="*", help="Optional list of file extensions to include (e.g. txt md)")
    p_ingest.add_argument("--output", default="stm_state.json", help="Path to output analysis state JSON file")
    p_ingest.add_argument("--store-signals", action="store_true", help="Persist per-window metrics in the output state")
    p_ingest.add_argument("--store-occurrences", action="store_true", help="Persist raw string occurrences in the output state")
    p_ingest.add_argument("--store-profiles", action="store_true", help="Persist aggregated string profiles in the output state")
    p_ingest.add_argument("--summary-top", dest="summary_top", type=int, default=10, help="Number of top items to include in the summary output")
    p_ingest.set_defaults(func=cmd_ingest)
    # Strings command
    p_strings = subparsers.add_parser("strings", help="List strings by patternability")
    p_strings.add_argument("--input", default="stm_state.json", help="Path to analysis state JSON file")
    p_strings.add_argument("--top", type=int, help="Number of top strings to display")
    p_strings.add_argument("--sort", choices=("pattern", "connector"), default="pattern", help="Sort key to use")
    p_strings.set_defaults(func=cmd_strings)
    # Themes command
    p_themes = subparsers.add_parser("themes", help="List detected themes")
    p_themes.add_argument("--input", default="stm_state.json", help="Path to analysis state JSON file")
    p_themes.set_defaults(func=cmd_themes)
    # Summary command
    p_summary = subparsers.add_parser("summary", help="Show aggregated analysis statistics")
    p_summary.add_argument("--input", default="stm_state.json", help="Path to analysis state JSON file")
    p_summary.add_argument("--top", type=int, help="Number of top strings/connectors to display")
    p_summary.set_defaults(func=cmd_summary)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
