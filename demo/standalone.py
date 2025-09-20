#!/usr/bin/env python3
"""Generate canned demo payloads for the Structural Intelligence Engine."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "analysis"
NOTE_DIR = REPO_ROOT / "docs" / "note"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "demo_payload.json"

_METRIC_KEYS = ("coherence", "stability", "entropy", "rupture", "lambda_hazard")


def _load_json(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artefact: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Artefact is empty: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_value(signal: Mapping[str, Any], key: str) -> float:
    metrics = signal.get("metrics") or {}
    value = metrics.get(key)
    if value is None:
        value = signal.get(key, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mean(values: Iterable[float]) -> float:
    seq = [float(v) for v in values if v is not None]
    if not seq:
        return 0.0
    try:
        return statistics.fmean(seq)
    except AttributeError:  # pragma: no cover - Python < 3.8 guard
        return statistics.mean(seq)


def _average_metrics(accumulator: Mapping[str, float], count: int) -> Dict[str, float]:
    if count <= 0:
        return {k: 0.0 for k in _METRIC_KEYS}
    return {k: accumulator.get(k, 0.0) / count for k in _METRIC_KEYS}


def _round_dict(data: Mapping[str, float], digits: int = 4) -> Dict[str, float]:
    return {k: round(float(v), digits) for k, v in data.items()}


def _normalise_signature(signature: str) -> str:
    if not signature:
        return "(unknown)"
    return signature.replace("__", "·")


def _summarise_samples(samples: Sequence[Mapping[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for sample in samples[:limit]:
        out.append({
            "index": int(sample.get("index", 0)),
            "coherence": round(float(sample.get("coherence", 0.0)), 4),
            "stability": round(float(sample.get("stability", 0.0)), 4),
            "entropy": round(float(sample.get("entropy", 0.0)), 4),
            "lambda_hazard": round(float(sample.get("lambda_hazard", 0.0)), 4),
        })
    return out


def analyse_pattern_prophet(state: Mapping[str, Any]) -> Dict[str, Any]:
    signals: Sequence[Mapping[str, Any]] = state.get("signals", [])  # type: ignore[assignment]
    total = len(signals)
    if total == 0:
        return {"error": "no signals available"}

    event_cut = max(10, int(total * 0.08))  # focus on the final ~8% of windows
    event_cut = min(event_cut, total)
    history_total = max(total - event_cut, 0)

    metrics_sum_event: MutableMapping[str, Dict[str, float]] = defaultdict(lambda: {k: 0.0 for k in _METRIC_KEYS})
    metrics_sum_hist: MutableMapping[str, Dict[str, float]] = defaultdict(lambda: {k: 0.0 for k in _METRIC_KEYS})
    samples_event: MutableMapping[str, List[Dict[str, float]]] = defaultdict(list)
    first_event_index: Dict[str, int] = {}
    last_event_index: Dict[str, int] = {}
    last_history_index: Dict[str, int] = {}

    event_counter: Counter[str] = Counter()
    history_counter: Counter[str] = Counter()

    # Accumulate history statistics first
    for signal in signals[:history_total]:
        signature = str(signal.get("signature", ""))
        idx = int(signal.get("index", signal.get("window_end", 0)))
        history_counter[signature] += 1
        last_history_index[signature] = idx
        bucket = metrics_sum_hist[signature]
        for key in _METRIC_KEYS:
            bucket[key] += _metric_value(signal, key)

    for signal in signals[history_total:]:
        signature = str(signal.get("signature", ""))
        idx = int(signal.get("index", signal.get("window_end", 0)))
        event_counter[signature] += 1
        bucket = metrics_sum_event[signature]
        for key in _METRIC_KEYS:
            bucket[key] += _metric_value(signal, key)
        if signature not in first_event_index:
            first_event_index[signature] = idx
        last_event_index[signature] = idx
        samples = samples_event[signature]
        if len(samples) < 5:
            samples.append({
                "index": idx,
                **{key: _metric_value(signal, key) for key in _METRIC_KEYS},
            })

    candidates: List[Dict[str, Any]] = []
    for signature, event_count in event_counter.items():
        hist_count = history_counter.get(signature, 0)
        total_count = event_count + hist_count
        event_rate = event_count / max(event_cut, 1)
        history_rate = hist_count / max(history_total, 1) if history_total else 0.0
        lift = (event_rate / history_rate) if history_rate > 0 else None
        candidate = {
            "signature": signature,
            "label": _normalise_signature(signature),
            "event_count": event_count,
            "history_count": hist_count,
            "total_count": total_count,
            "event_rate": round(event_rate, 6),
            "history_rate": round(history_rate, 6),
            "lift": round(lift, 3) if lift is not None else None,
            "event_metrics": _round_dict(_average_metrics(metrics_sum_event[signature], event_count)),
            "history_metrics": _round_dict(_average_metrics(metrics_sum_hist.get(signature, {}), hist_count)),
            "history_last_index": last_history_index.get(signature),
            "event_span": {
                "first_index": first_event_index.get(signature),
                "last_index": last_event_index.get(signature),
            },
            "samples": _summarise_samples(samples_event[signature]),
        }
        if candidate["event_span"]["first_index"] is not None and candidate["history_last_index"] is not None:
            gap = candidate["event_span"]["first_index"] - candidate["history_last_index"]
            candidate["history_gap_windows"] = int(gap)
        candidates.append(candidate)

    candidates.sort(
        key=lambda c: (
            1 if c["history_count"] else 0,
            c["lift"] if c["lift"] is not None else 0.0,
            c["event_count"],
            c["history_count"],
        ),
        reverse=True,
    )
    selected = candidates[0] if candidates else None

    return {
        "total_windows": total,
        "event_window": event_cut,
        "coverage": round(event_cut / total, 4) if total else 0.0,
        "selected": selected,
        "candidates": candidates[:10],
    }


def analyse_twin_finder(twin_payload: Mapping[str, Any], *, limit: int = 3) -> Dict[str, Any]:
    results = twin_payload.get("results", [])
    summaries: List[Dict[str, Any]] = []
    for result in results:
        matches: Sequence[Mapping[str, Any]] = result.get("matches", [])  # type: ignore[assignment]
        if not matches:
            continue
        distances = [float(m.get("distance", 0.0)) for m in matches]
        metrics_coh = [float(m.get("metrics", {}).get("coherence", 0.0)) for m in matches]
        metrics_stab = [float(m.get("metrics", {}).get("stability", 0.0)) for m in matches]
        summaries.append({
            "string": result.get("string"),
            "label": result.get("string"),
            "source_occurrences": result.get("occurrences", 0),
            "patternability": round(float(result.get("patternability", 0.0)), 4),
            "connector": round(float(result.get("connector", 0.0)), 4),
            "twin_windows": len(matches),
            "mean_distance": round(_mean(distances), 6),
            "min_distance": round(min(distances), 6),
            "max_distance": round(max(distances), 6),
            "mean_coherence": round(_mean(metrics_coh), 4),
            "mean_stability": round(_mean(metrics_stab), 4),
            "sample_matches": [
                {
                    "window_id": int(match.get("window_id", 0)),
                    "distance": round(float(match.get("distance", 0.0)), 6),
                    "signature": match.get("signature"),
                    "window_start": int(match.get("window_start", 0)),
                    "window_end": int(match.get("window_end", 0)),
                }
                for match in matches[:5]
            ],
        })
    summaries.sort(key=lambda s: (s["twin_windows"], -s["mean_distance"]), reverse=True)
    return {
        "total_candidates": len(results),
        "top_matches": summaries[:limit],
    }


def analyse_context_refinery(state: Mapping[str, Any], proposals: Mapping[str, Any]) -> Dict[str, Any]:
    strings: Mapping[str, Mapping[str, Any]] = state.get("string_scores", {})  # type: ignore[assignment]
    total_occ = 0
    structural: List[Dict[str, Any]] = []
    non_structural: List[Dict[str, Any]] = []

    for text, payload in strings.items():
        occurrences = int(payload.get("occurrences", 0))
        total_occ += occurrences
        metrics = payload.get("metrics", {})
        entry = {
            "string": text,
            "occurrences": occurrences,
            "coherence": round(float(metrics.get("coherence", 0.0)), 4),
            "stability": round(float(metrics.get("stability", 0.0)), 4),
            "entropy": round(float(metrics.get("entropy", 0.0)), 4),
            "rupture": round(float(metrics.get("rupture", 0.0)), 4),
        }
        if "__" in text:
            structural.append(entry)
        else:
            non_structural.append(entry)

    structural.sort(key=lambda e: e["occurrences"], reverse=True)
    non_structural.sort(key=lambda e: e["occurrences"], reverse=True)

    proposal_items = proposals.get("proposals", [])
    ranked_proposals: List[Dict[str, Any]] = []
    structural_from_proposals: List[Dict[str, Any]] = []
    total_proposal_occ = 0
    structural_proposal_occ = 0
    for item in proposal_items:
        occurrences = int(item.get("occurrences", 0))
        total_proposal_occ += occurrences
        ranked_proposals.append({
            "string": item.get("string"),
            "score": round(float(item.get("score", 0.0)), 6),
            "patternability": round(float(item.get("patternability", 0.0)), 4),
            "connector": round(float(item.get("connector", 0.0)), 4),
            "occurrences": occurrences,
            "coherence": round(float(item.get("diagnostics", {}).get("coherence", 0.0)), 4),
            "stability": round(float(item.get("diagnostics", {}).get("stability", 0.0)), 4),
            "entropy": round(float(item.get("diagnostics", {}).get("entropy", 0.0)), 4),
            "rupture": round(float(item.get("diagnostics", {}).get("rupture", 0.0)), 4),
        })
        if "__" in str(item.get("string", "")):
            structural_proposal_occ += occurrences
            structural_from_proposals.append(ranked_proposals[-1])
    ranked_proposals.sort(key=lambda e: (e["score"], e["connector"]), reverse=True)

    structural_from_proposals.sort(key=lambda e: e["occurrences"], reverse=True)
    structural_share = (structural_proposal_occ / total_proposal_occ) if total_proposal_occ else 0.0

    return {
        "total_strings": len(strings),
        "structural_strings": len(structural_from_proposals),
        "structural_share": round(structural_share, 4),
        "top_structural": structural_from_proposals[:8] or structural[:8],
        "top_non_structural": non_structural[:5],
        "top_proposals": ranked_proposals[:8],
    }


def build_payload() -> Dict[str, Any]:
    state_path = ANALYSIS_DIR / "mms_state.json"
    proposals_path = ANALYSIS_DIR / "mms_proposals_struct.json"
    twins_path = ANALYSIS_DIR / "mms_twins_2017-09-07_to_13.json"

    state = _load_json(state_path)
    proposals = _load_json(proposals_path)
    twins = _load_json(twins_path)

    pattern_prophet = analyse_pattern_prophet(state)
    twin_finder = analyse_twin_finder(twins)
    context_refinery = analyse_context_refinery(state, proposals)

    assets = {
        "pattern_prophet": str((NOTE_DIR / "fig3_mms_0000_lead.png").relative_to(REPO_ROOT) if (NOTE_DIR / "fig3_mms_0000_lead.png").exists() else ""),
        "twin_finder": str((NOTE_DIR / "fig2_mms_0000_zoom.png").relative_to(REPO_ROOT) if (NOTE_DIR / "fig2_mms_0000_zoom.png").exists() else ""),
        "context_refinery": str((NOTE_DIR / "fig1_mms_0000_overview.png").relative_to(REPO_ROOT) if (NOTE_DIR / "fig1_mms_0000_overview.png").exists() else ""),
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "state": str(state_path.relative_to(REPO_ROOT)),
            "proposals": str(proposals_path.relative_to(REPO_ROOT)),
            "twins": str(twins_path.relative_to(REPO_ROOT)),
        },
        "assets": assets,
        "demos": {
            "pattern_prophet": pattern_prophet,
            "twin_finder": twin_finder,
            "context_refinery": context_refinery,
        },
    }


def _print_summary(payload: Mapping[str, Any]) -> None:
    demos = payload.get("demos", {})
    pp = demos.get("pattern_prophet", {})
    selected = pp.get("selected") or {}
    sig = selected.get("label") or selected.get("signature") or "(none)"
    print(f"Pattern Prophet · signature {sig} — events: {selected.get('event_count', 0)} | lift: {selected.get('lift')}")

    tf = demos.get("twin_finder", {})
    top_matches = tf.get("top_matches", [])
    if top_matches:
        tm = top_matches[0]
        print(f"Twin Finder · {tm.get('string')} with {tm.get('twin_windows')} aligned windows (mean distance {tm.get('mean_distance')})")
    else:
        print("Twin Finder · no matches available")

    cr = demos.get("context_refinery", {})
    share = cr.get("structural_share")
    print(f"Context Refinery · structural share: {share}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the generated payload JSON.",
    )
    parser.add_argument(
        "--no-pretty",
        action="store_true",
        help="Disable pretty printing (compact JSON).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.no_pretty:
        serialized = json.dumps(payload, separators=(",", ":"))
    else:
        serialized = json.dumps(payload, indent=2, sort_keys=True)
    args.output.write_text(serialized + ("\n" if not args.no_pretty else ""), encoding="utf-8")
    print(f"Demo payload written to {args.output}")
    _print_summary(payload)


if __name__ == "__main__":
    main()
