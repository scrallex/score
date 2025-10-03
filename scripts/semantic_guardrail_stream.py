#!/usr/bin/env python3
"""Generate a scripted event stream for the Semantic Guardrail demo.

The script simulates a sequence of events drawn from one or more STM
state files.  For each string we compute its structural patternability
and semantic similarity to a seed list, then emit JSON Lines showing how
naïve semantic or structural guardrails behave versus the hybrid
guardrail.

Example
-------

    PYTHONPATH=src .venv/bin/python scripts/semantic_guardrail_stream.py \
        --seeds risk resilience volatility anomaly "predictive maintenance"

Outputs `results/semantic_guardrail_stream.jsonl` and a short summary
table describing the mix of semantic-only, structural-only, neutral, and
hybrid-alert events.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from sep_text_manifold.semantic import EmbeddingConfig, SemanticEmbedder, seed_similarity


DEFAULT_STATES: List[Tuple[str, Path]] = [
    ("docs", Path("analysis/semantic_demo_state.json")),
    ("mms", Path("analysis/mms_state.json")),
]


@dataclass
class EventCandidate:
    source: str
    string: str
    patternability: float
    semantic_similarity: float
    coherence: float
    stability: float
    entropy: float
    rupture: float
    occurrences: int


def parse_state_argument(arg: str) -> Tuple[str, Path]:
    if "=" not in arg:
        raise argparse.ArgumentTypeError("State must be provided as label=path")
    label, path = arg.split("=", 1)
    return label.strip(), Path(path.strip())


def load_candidates(
    states: Iterable[Tuple[str, Path]],
    seeds: List[str],
    embedder: SemanticEmbedder,
) -> List[EventCandidate]:
    candidates: List[EventCandidate] = []
    for label, path in states:
        data = json.loads(path.read_text())
        string_scores: Dict[str, Dict[str, object]] = data.get("string_scores", {})
        strings = list(string_scores.keys())
        if not strings:
            continue
        sem = seed_similarity(strings, embedder=embedder, seeds=seeds)
        for idx, string in enumerate(strings):
            entry = string_scores[string]
            metrics = entry.get("metrics", {})
            pattern = float(entry.get("patternability", metrics.get("coherence", 0.0)))
            coherence = float(entry.get("coherence", metrics.get("coherence", 0.0)))
            stability = float(entry.get("stability", metrics.get("stability", 0.0)))
            entropy = float(entry.get("entropy", metrics.get("entropy", 0.0)))
            rupture = float(entry.get("rupture", metrics.get("rupture", 0.0)))
            occurrences = int(entry.get("occurrences", 0))
            candidates.append(
                EventCandidate(
                    source=label,
                    string=string,
                    patternability=pattern,
                    semantic_similarity=float(sem[idx]),
                    coherence=coherence,
                    stability=stability,
                    entropy=entropy,
                    rupture=rupture,
                    occurrences=occurrences,
                )
            )
    return candidates


def choose_candidates(
    rng: random.Random,
    population: List[EventCandidate],
    limit: int,
) -> List[EventCandidate]:
    if not population:
        return []
    if limit >= len(population):
        return population[:]
    idxs = rng.sample(range(len(population)), limit)
    return [population[i] for i in idxs]


def build_timeline(
    rng: random.Random,
    semantic_only: List[EventCandidate],
    structural_only: List[EventCandidate],
    neutral: List[EventCandidate],
    incident: EventCandidate,
    *,
    repetitions: int,
) -> List[EventCandidate]:
    timeline: List[EventCandidate] = []
    seq_sem = choose_candidates(rng, semantic_only, repetitions)
    seq_struct = choose_candidates(rng, structural_only, repetitions)
    seq_neutral = choose_candidates(rng, neutral, max(1, repetitions // 2))

    # Interleave: structural, semantic, neutral … then inject incident near the end.
    pairs = max(len(seq_sem), len(seq_struct))
    for idx in range(pairs):
        if idx < len(seq_struct):
            timeline.append(seq_struct[idx])
        if idx < len(seq_sem):
            timeline.append(seq_sem[idx])
        if idx < len(seq_neutral):
            timeline.append(seq_neutral[idx])
    # Insert incident towards the end.
    insert_at = max(0, len(timeline) - max(1, len(timeline) // 4))
    timeline.insert(insert_at, incident)
    return timeline


def event_record(
    step: int,
    candidate: EventCandidate,
    semantic_threshold: float,
    structural_threshold: float,
    *,
    cluster: str,
    top_pool: Sequence[EventCandidate],
    rng: random.Random,
) -> Dict[str, object]:
    naive_semantic = candidate.semantic_similarity >= semantic_threshold
    naive_structural = candidate.patternability >= structural_threshold
    hybrid = naive_semantic and naive_structural

    twins = build_twins(candidate, top_pool)
    repair_suggestion: Optional[Dict[str, object]] = None
    repair_applied = False
    if not hybrid and twins:
        repair_suggestion = dict(twins[0])
        repair_applied = True

    latency_ms = round(rng.uniform(35.0, 85.0), 2)

    return {
        "step": step,
        "source": candidate.source,
        "event": candidate.string,
        "cluster": cluster,
        "patternability": round(candidate.patternability, 6),
        "semantic_similarity": round(candidate.semantic_similarity, 6),
        "coherence": round(candidate.coherence, 6),
        "stability": round(candidate.stability, 6),
        "entropy": round(candidate.entropy, 6),
        "rupture": round(candidate.rupture, 6),
        "occurrences": candidate.occurrences,
        "naive_semantic_alert": naive_semantic,
        "naive_structural_alert": naive_structural,
        "hybrid_guardrail_alert": hybrid,
        "twins": twins,
        "repair_suggestion": repair_suggestion,
        "repair_applied": repair_applied,
        "latency_ms": latency_ms,
    }


def build_twins(
    candidate: EventCandidate,
    pool: Sequence[EventCandidate],
    *,
    limit: int = 3,
) -> List[Dict[str, object]]:
    """Return representative precedent windows for citation or repair."""

    def payload(item: EventCandidate) -> Dict[str, object]:
        return {
            "string": item.string,
            "source": item.source,
            "occurrences": item.occurrences,
            "patternability": round(item.patternability, 6),
            "semantic_similarity": round(item.semantic_similarity, 6),
        }

    results: List[Dict[str, object]] = []

    if candidate.occurrences >= 2:
        results.append(payload(candidate))

    for item in pool:
        if item.string == candidate.string:
            continue
        if item.occurrences < 2:
            break
        results.append(payload(item))
        if len(results) >= limit:
            break

    return results


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", required=True, help="Semantic seed phrases")
    parser.add_argument(
        "--state",
        action="append",
        type=parse_state_argument,
        help="State file in label=path form (defaults to docs and mms)",
    )
    parser.add_argument("--semantic-threshold", type=float, default=0.28)
    parser.add_argument("--structural-threshold", type=float, default=0.475)
    parser.add_argument("--samples", type=int, default=8, help="Per-cluster sample count")
    parser.add_argument("--output", type=Path, default=Path("results/semantic_guardrail_stream.jsonl"))
    parser.add_argument("--metrics-output", type=Path, help="Optional path for aggregate metrics JSON")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--incident-name", default="database_connection_timeout")
    parser.add_argument("--incident-source", default="synthetic")
    parser.add_argument("--incident-pattern", type=float, default=0.478)
    parser.add_argument("--incident-semantic", type=float, default=0.32)
    parser.add_argument("--incident-coherence", type=float, default=0.07)
    parser.add_argument("--incident-stability", type=float, default=0.6)
    parser.add_argument("--incident-entropy", type=float, default=0.93)
    parser.add_argument("--incident-rupture", type=float, default=0.4)
    parser.add_argument("--incident-occurrences", type=int, default=4)

    args = parser.parse_args(argv)

    states = args.state or DEFAULT_STATES
    embedder = SemanticEmbedder(EmbeddingConfig(method="transformer"))
    candidates = load_candidates(states, args.seeds, embedder)
    if not candidates:
        parser.error("No strings found in the provided states")

    sem_thr = args.semantic_threshold
    struct_thr = args.structural_threshold

    semantic_only = [
        c for c in candidates if c.semantic_similarity >= sem_thr and c.patternability < struct_thr * 0.985
    ]
    structural_only = [
        c for c in candidates if c.patternability >= struct_thr and c.semantic_similarity < sem_thr * 0.75
    ]
    neutral = [
        c for c in candidates if c not in semantic_only and c not in structural_only
    ]

    incident_candidate = EventCandidate(
        source=args.incident_source,
        string=args.incident_name,
        patternability=args.incident_pattern,
        semantic_similarity=args.incident_semantic,
        coherence=args.incident_coherence,
        stability=args.incident_stability,
        entropy=args.incident_entropy,
        rupture=args.incident_rupture,
        occurrences=args.incident_occurrences,
    )

    rng = random.Random(args.seed)
    top_pool = sorted(candidates + [incident_candidate], key=lambda c: c.occurrences, reverse=True)
    timeline = build_timeline(
        rng,
        semantic_only,
        structural_only,
        neutral,
        incident_candidate,
        repetitions=args.samples,
    )

    clusters = {
        id(c): "semantic_only" for c in semantic_only
    }
    clusters.update({id(c): "structural_only" for c in structural_only})
    clusters.update({id(c): "neutral" for c in neutral})
    clusters[id(incident_candidate)] = "hybrid_alert"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = {"semantic_only": 0, "structural_only": 0, "neutral": 0, "hybrid_alert": 0}
    approved = 0
    blocked = 0
    repairs = 0
    citations = 0
    latency_samples: List[float] = []
    semantic_trips = 0
    structural_trips = 0
    with args.output.open("w", encoding="utf-8") as fh:
        for idx, candidate in enumerate(timeline):
            cluster = clusters.get(id(candidate), "neutral")
            record = event_record(
                idx,
                candidate,
                sem_thr,
                struct_thr,
                cluster=cluster,
                top_pool=top_pool,
                rng=rng,
            )
            if candidate is incident_candidate:
                record.update({"annotation": "synthetic incident"})
            fh.write(json.dumps(record) + "\n")
            summary[cluster] = summary.get(cluster, 0) + 1
            if record.get("naive_semantic_alert"):
                semantic_trips += 1
            if record.get("naive_structural_alert"):
                structural_trips += 1
            if record.get("hybrid_guardrail_alert"):
                approved += 1
                if record.get("twins"):
                    citations += 1
            else:
                blocked += 1
                if record.get("repair_applied"):
                    repairs += 1
            latency_samples.append(float(record.get("latency_ms", 0.0)))

    print(f"Wrote {len(timeline)} events to {args.output}")
    print("Event mix:")
    for key, value in summary.items():
        print(f"  {key:16s}: {value}")

    if args.metrics_output:
        total = approved + blocked
        hall_rate = blocked / total if total else 0.0
        repair_yield = repairs / blocked if blocked else 0.0
        citation_coverage = citations / approved if approved else 0.0
        latency_mean = float(np.mean(latency_samples)) if latency_samples else 0.0
        latency_p95 = float(np.percentile(latency_samples, 95)) if latency_samples else 0.0
        metrics_payload = {
            "total_events": total,
            "semantic_alerts": semantic_trips,
            "structural_alerts": structural_trips,
            "hybrid_alerts": approved,
            "blocked": blocked,
            "repairs": repairs,
            "citations": citations,
            "hallucination_rate": hall_rate,
            "repair_yield": repair_yield,
            "citation_coverage": citation_coverage,
            "latency_ms_mean": latency_mean,
            "latency_ms_p95": latency_p95,
        }
        metrics_path = args.metrics_output
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics_payload, indent=2))
        print(f"Wrote metrics summary to {metrics_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
