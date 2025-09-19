"""CLI for auto-labelling onsets."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from .rules import MMS_RULES


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


def detect_onsets(state: Dict, thresholds: Dict[str, float], times: np.ndarray) -> List[Dict[str, str]]:
    signals = state.get("signals", [])
    onsets: List[Dict[str, str]] = []
    above = False
    for idx, sig in enumerate(signals):
        metrics = sig.get("metrics", {})
        coh = metrics.get("coherence", 0.0)
        ent = metrics.get("entropy", 1.0)
        stab = metrics.get("stability", 0.0)
        meets = (
            coh >= thresholds["min_coh"]
            and ent <= thresholds["max_ent"]
            and stab >= thresholds["min_stab"]
        )
        if meets:
            if not above:
                ts = times[idx].isoformat()
                onsets.append({"onset": ts, "confidence": 0.6, "rule": "guardrail"})
                above = True
        else:
            above = False
    return onsets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-label onsets using router thresholds")
    parser.add_argument("--state", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--stop", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = load_state(Path(args.state))
    thresholds = load_thresholds(Path(args.config))
    start = datetime.fromisoformat(args.start)
    stop = datetime.fromisoformat(args.stop)
    times = build_timeline(len(state.get("signals", [])), start, stop)
    onsets = detect_onsets(state, thresholds, times)
    output = Path(args.output) if args.output else Path(args.state).with_name("onsets.json")
    output.write_text(json.dumps(onsets, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
