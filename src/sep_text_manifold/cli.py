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
import os
from pathlib import Path
from typing import Dict, Any, List

from .ingest import ingest_directory
from .manifold import build_manifold
from .strings import extract_strings, aggregate_string_metrics
from .scoring import patternability_score
from .themes import build_theme_graph, detect_themes, compute_graph_metrics


def _load_state(state_file: Path) -> Dict[str, Any]:
    if state_file.exists():
        with state_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_state(state_file: Path, data: Dict[str, Any]) -> None:
    with state_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest a directory of text files and analyse it."""
    directory = args.directory
    window_bytes = args.window_bytes
    stride = args.stride
    state_file = Path(args.output)
    # Read all files and concatenate their bytes with a separator
    corpus_bytes = bytearray()
    occurrences = []
    # Keep track of byte offset for each file to realign occurrences
    current_offset = 0
    for file_id, path, text in ingest_directory(directory, extensions=args.extensions):
        # Extract occurrences for this file
        occs = extract_strings(text, file_id)
        # Adjust byte offsets by current_offset
        for occ in occs:
            occ.byte_start += current_offset
            occ.byte_end += current_offset
            occurrences.append(occ)
        # Append file bytes and a newline separator
        data_bytes = text.encode("utf-8")
        corpus_bytes.extend(data_bytes)
        # Use a single byte separator to delineate files
        corpus_bytes.append(0)
        current_offset += len(data_bytes) + 1
    # Build manifold
    signals = build_manifold(bytes(corpus_bytes), window_bytes=window_bytes, stride=stride)
    # Aggregate string metrics
    string_profiles = aggregate_string_metrics(
        occurrences,
        signals,
        window_bytes=window_bytes,
        stride=stride,
    )
    # Compute patternability score for each string
    string_scores: Dict[str, Dict[str, Any]] = {}
    for s, profile in string_profiles.items():
        metrics = profile.get("metrics", {})
        p_score = patternability_score(
            metrics.get("coherence", 0.0),
            metrics.get("stability", 0.0),
            metrics.get("entropy", 0.0),
            metrics.get("rupture", 0.0),
        )
        string_scores[s] = {
            "metrics": metrics,
            "occurrences": profile.get("occurrences", 0),
            "window_ids": profile.get("window_ids", []),
            "patternability": p_score,
        }
        for field, value in metrics.items():
            string_scores[s][field] = value
    # Build co‑occurrence graph using shared manifold windows per string
    graph = build_theme_graph({s: prof.get("window_ids", []) for s, prof in string_scores.items()})
    themes = detect_themes(graph)
    graph_metrics = compute_graph_metrics(graph)
    # Combine connector metrics into string_scores
    from .scoring import connector_score
    for s in string_scores:
        gm = graph_metrics.get(s, {})
        c_score = connector_score(
            gm.get("betweenness", 0.0),
            gm.get("bridging", 0.0),
            0.0,  # PMI across themes not computed yet
            gm.get("theme_entropy_neighbors", 0.0),
            gm.get("redundant_degree", 0.0),
        )
        string_scores[s]["connector"] = c_score
        string_scores[s]["graph_metrics"] = gm
    # Save state
    state = {
        "string_scores": string_scores,
        "themes": [list(t) for t in themes],
    }
    _save_state(state_file, state)
    print(f"Analysis complete.  Saved state to {state_file}")


def cmd_strings(args: argparse.Namespace) -> None:
    """List strings sorted by patternability."""
    state = _load_state(Path(args.input))
    scores: Dict[str, Dict[str, Any]] = state.get("string_scores", {})
    if not scores:
        print("No string scores found.  Run 'stm ingest' first.")
        return
    # Sort by patternability descending
    sorted_items = sorted(scores.items(), key=lambda kv: kv[1].get("patternability", 0.0), reverse=True)
    top = args.top or len(sorted_items)
    for i, (s, metrics) in enumerate(sorted_items[:top], start=1):
        p_score = metrics.get("patternability", 0.0)
        print(f"{i:3d}. {s:20s} {p_score:.3f}")


def cmd_themes(args: argparse.Namespace) -> None:
    """List themes and their member strings."""
    state = _load_state(Path(args.input))
    themes: List[List[str]] = state.get("themes", [])
    if not themes:
        print("No themes found.  Run 'stm ingest' first.")
        return
    for i, members in enumerate(themes, start=1):
        print(f"Theme {i} ({len(members)} strings): {', '.join(sorted(members)[:20])}{'...' if len(members) > 20 else ''}")


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
    p_ingest.set_defaults(func=cmd_ingest)
    # Strings command
    p_strings = subparsers.add_parser("strings", help="List strings by patternability")
    p_strings.add_argument("--input", default="stm_state.json", help="Path to analysis state JSON file")
    p_strings.add_argument("--top", type=int, help="Number of top strings to display")
    p_strings.set_defaults(func=cmd_strings)
    # Themes command
    p_themes = subparsers.add_parser("themes", help="List detected themes")
    p_themes.add_argument("--input", default="stm_state.json", help="Path to analysis state JSON file")
    p_themes.set_defaults(func=cmd_themes)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
