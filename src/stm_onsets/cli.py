"""CLI for auto-labelling onsets."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from sep_text_manifold.cli_lead import main as lead_main
from .rules import evaluate_mms_rules, score_hits


def load_state(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_thresholds(path: Path) -> Dict[str, float]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    foreground = cfg["router"]["foreground"]
    return {
        "min_coh": foreground.get("min_coh", 0.0),
        "max_ent": foreground.get("max_ent", 1.0),
        "min_stab": foreground.get("min_stab", 0.0),
    }


def build_timeline(count: int, start: datetime, stop: datetime) -> np.ndarray:
    if count == 0:
        return np.array([])
    delta = (stop - start) / count
    return np.array([start + (i + 0.5) * delta for i in range(count)])


def detect_onsets(
    state: Dict,
    thresholds: Dict[str, float],
    times: np.ndarray,
    streak: int,
) -> List[Dict[str, object]]:
    signals = state.get("signals", [])
    onsets: List[Dict[str, object]] = []
    prev_metrics: Dict[str, float] | None = None
    streak_count = 0
    for idx, sig in enumerate(signals):
        metrics = sig.get("metrics", {})
        hits = evaluate_mms_rules(prev_metrics, metrics, thresholds)
        score = score_hits(hits)
        guard = any(hit.name == "guardrail" for hit in hits)
        if guard:
            streak_count += 1
        else:
            streak_count = 0
        if streak_count >= streak:
            onset_idx = idx - streak + 1
            ts = times[onset_idx].isoformat()
            onsets.append(
                {
                    "onset": ts,
                    "confidence": min(1.0, score + 0.2 * (streak - 1)),
                    "evidence": [hit.name for hit in hits],
                    "window_index": onset_idx,
                }
            )
            streak_count = 0
        prev_metrics = metrics
    return onsets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-label onsets using router thresholds")
    parser.add_argument("--state", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--stop", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--streak", type=int, default=3, help="Consecutive windows required to confirm onset")
    parser.add_argument("--lead-output", dest="lead_output", help="Optional lead-time JSON output path")
    parser.add_argument("--lead-plot", dest="lead_plot", help="Optional lead-time plot path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = load_state(Path(args.state))
    thresholds = load_thresholds(Path(args.config))
    start = datetime.fromisoformat(args.start)
    stop = datetime.fromisoformat(args.stop)
    times = build_timeline(len(state.get("signals", [])), start, stop)
    onsets = detect_onsets(state, thresholds, times, args.streak)
    output = Path(args.output) if args.output else Path(args.state).with_name("onsets.json")
    output.write_text(json.dumps(onsets, indent=2), encoding="utf-8")
    print(f"wrote {output}")
    if onsets and (args.lead_output or args.lead_plot):
        onset_ts = onsets[0]["onset"]
        argv = [
            "stm-leadtime",
            "--state",
            args.state,
            "--onset",
            onset_ts,
        ]
        if args.lead_output:
            argv.extend(["--output", args.lead_output])
        if args.lead_plot:
            argv.extend(["--plot", args.lead_plot])
        old_argv = sys.argv
        sys.argv = argv
        try:
            lead_main()
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    main()
