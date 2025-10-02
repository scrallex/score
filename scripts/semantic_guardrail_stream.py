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
from typing import Dict, Iterable, List, Tuple

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
) -> Dict[str, object]:
    naive_semantic = candidate.semantic_similarity >= semantic_threshold
    naive_structural = candidate.patternability >= structural_threshold
    hybrid = naive_semantic and naive_structural
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
    }


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
    with args.output.open("w", encoding="utf-8") as fh:
        for idx, candidate in enumerate(timeline):
            cluster = clusters.get(id(candidate), "neutral")
            record = event_record(
                idx,
                candidate,
                sem_thr,
                struct_thr,
                cluster=cluster,
            )
            if candidate is incident_candidate:
                record.update({"annotation": "synthetic incident"})
            fh.write(json.dumps(record) + "\n")
            summary[cluster] = summary.get(cluster, 0) + 1

    print(f"Wrote {len(timeline)} events to {args.output}")
    print("Event mix:")
    for key, value in summary.items():
        print(f"  {key:16s}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

