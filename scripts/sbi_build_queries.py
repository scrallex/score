#!/usr/bin/env python3
"""Build SBI experiment query sets for membership, twins, and context ranking."""

from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from bloom_filter2 import BloomFilter
from rapidfuzz.distance import Levenshtein

from sbi.span_corpus import (
    ContextAccumulator,
    SpanAggregate,
    build_span_aggregates,
    compute_span_id,
    load_span_inventory,
    write_context_table,
    write_span_inventory,
)

POSITIVE_COUNT = 10_000
STRUCT_COUNT = 10_000
SEMANTIC_COUNT = 10_000
CONTEXT_COUNT = 5_000

DEFAULT_SEED = 1729


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("analysis/truth_packs/example/manifest.json"),
        help="Truth-pack manifest to source spans from.",
    )
    parser.add_argument(
        "--span-inventory",
        type=Path,
        help="Optional path to reuse/store span inventory JSONL (defaults under pack).",
    )
    parser.add_argument(
        "--dev-jsonl",
        type=Path,
        default=Path("data/corpus_example/claims.jsonl"),
        help="JSONL corpus of free-form claims used to sample negatives.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/sbi"),
        help="Directory to write query JSONL files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuilding the span inventory even if cached file exists.",
    )
    return parser.parse_args()


def ensure_span_inventory(
    manifest_path: Path,
    inventory_path: Optional[Path],
    rebuild: bool,
) -> Tuple[Dict[str, SpanAggregate], Path]:
    if inventory_path is None:
        inventory_path = manifest_path.parent / "sbi" / "spans.jsonl"
    if inventory_path.exists() and not rebuild:
        return load_span_inventory(inventory_path), inventory_path
    aggregates = build_span_aggregates(manifest_path)
    write_span_inventory(aggregates, output_path=inventory_path)
    return aggregates, inventory_path


def sample_positive(spans: Sequence[SpanAggregate], *, count: int, rng: random.Random) -> List[Dict[str, object]]:
    if len(spans) < count:
        raise ValueError(f"Not enough spans to sample {count} positives (have {len(spans)})")
    sample = rng.sample(spans, count)
    records: List[Dict[str, object]] = []
    for span in sample:
        sources = sorted(span.sources.items(), key=lambda item: (-item[1], item[0]))[:5]
        records.append(
            {
                "span_id": span.span_id,
                "text": span.text,
                "occurrences": span.occurrences,
                "sources": [
                    {
                        "record_id": record_id,
                        "count": count,
                    }
                    for record_id, count in sources
                ],
            }
        )
    return records


def read_dev_claims(path: Path) -> List[str]:
    claims: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            claim = payload.get("claim")
            if isinstance(claim, str) and claim.strip():
                claims.append(claim.strip())
    return claims


def sample_negatives(
    *,
    claims: Sequence[str],
    span_ids: set[str],
    count: int,
    rng: random.Random,
) -> List[Dict[str, object]]:
    candidates: List[str] = []
    seen_ids: set[str] = set()
    for claim in claims:
        span_id = compute_span_id(claim)
        if span_id in span_ids or span_id in seen_ids:
            continue
        candidates.append(claim)
        seen_ids.add(span_id)
    if len(candidates) < count:
        raise ValueError(f"Not enough negative candidates (need {count}, have {len(candidates)})")
    sample = rng.sample(candidates, count)
    return [{"text": text, "span_id": compute_span_id(text)} for text in sample]


def mutate_span(text: str, rng: random.Random, max_attempts: int = 20) -> Tuple[str, int]:
    if not text:
        raise ValueError("Cannot mutate empty text")
    operations = ["delete", "insert", "substitute", "swap", "word_swap"]
    letters = "abcdefghijklmnopqrstuvwxyz"
    words = text.split()
    for _ in range(max_attempts):
        op = rng.choice(operations)
        chars = list(text)
        if op == "delete" and len(chars) > 1:
            idx = rng.randrange(len(chars))
            mutated = text[:idx] + text[idx + 1 :]
        elif op == "insert":
            idx = rng.randrange(len(chars) + 1)
            ch = rng.choice(letters)
            mutated = text[:idx] + ch + text[idx:]
        elif op == "substitute" and len(chars) > 0:
            idx = rng.randrange(len(chars))
            ch = rng.choice(letters)
            mutated = text[:idx] + ch + text[idx + 1 :]
        elif op == "swap" and len(chars) >= 2:
            idx = rng.randrange(len(chars) - 1)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            mutated = "".join(chars)
        elif op == "word_swap" and len(words) >= 2:
            i, j = rng.sample(range(len(words)), 2)
            words_copy = words[:]
            words_copy[i], words_copy[j] = words_copy[j], words_copy[i]
            mutated = " ".join(words_copy)
        else:
            continue
        mutated = mutated.strip()
        if not mutated or mutated == text:
            continue
        distance = Levenshtein.distance(text, mutated)
        if 1 <= distance <= 3:
            return mutated, distance
    raise RuntimeError("Failed to generate valid structural twin")


def build_structural_twins(
    spans: Sequence[SpanAggregate],
    *,
    count: int,
    rng: random.Random,
) -> List[Dict[str, object]]:
    if len(spans) < count:
        raise ValueError("Insufficient spans for structural twins")
    sample = rng.sample(spans, count)
    records: List[Dict[str, object]] = []
    for span in sample:
        mutated, distance = mutate_span(span.text, rng)
        records.append(
            {
                "query": mutated,
                "target_span": span.text,
                "span_id": span.span_id,
                "edit_distance": distance,
            }
        )
    return records


SEMANTIC_TEMPLATES = [
    "According to the evidence, {text}",
    "It has been reported that {text}",
    "Sources note that {text}",
    "The records indicate that {text}",
    "Dataset documentation states that {text}",
    "As summarised in the corpus, {text}",
]


def paraphrase_span(text: str, rng: random.Random) -> str:
    template = rng.choice(SEMANTIC_TEMPLATES)
    return template.format(text=text.rstrip(".")) + "."


def build_semantic_twins(
    spans: Sequence[SpanAggregate],
    *,
    count: int,
    rng: random.Random,
) -> List[Dict[str, object]]:
    if len(spans) < count:
        raise ValueError("Insufficient spans for semantic twins")
    sample = rng.sample(spans, count)
    records: List[Dict[str, object]] = []
    for span in sample:
        query = paraphrase_span(span.text, rng)
        records.append(
            {
                "query": query,
                "targets": [
                    {
                        "span_id": span.span_id,
                        "text": span.text,
                        "occurrences": span.occurrences,
                    }
                ],
            }
        )
    return records


def build_context_queries(
    spans: Sequence[SpanAggregate],
    *,
    count: int,
    rng: random.Random,
) -> List[Dict[str, object]]:
    candidates: List[SpanAggregate] = []
    for span in spans:
        if span.left_contexts and span.right_contexts:
            candidates.append(span)
    if len(candidates) < count:
        raise ValueError(f"Not enough spans with contexts (need {count}, have {len(candidates)})")
    sample = rng.sample(candidates, count)
    records: List[Dict[str, object]] = []
    for span in sample:
        def serialise(contexts: Dict[str, ContextAccumulator]) -> List[Dict[str, object]]:
            items = sorted(contexts.items(), key=lambda item: (item[1].count, item[0]), reverse=True)
            serialised: List[Dict[str, object]] = []
            for text, acc in items:
                serialised.append(
                    {
                        "text": text,
                        "count": acc.count,
                        "unique_uris": acc.unique_uris,
                        "unique_domains": acc.unique_domains,
                        "unique_timebins": acc.unique_timebins,
                    }
                )
            return serialised

        records.append(
            {
                "span_id": span.span_id,
                "span": span.text,
                "left": serialise(span.left_contexts),
                "right": serialise(span.right_contexts),
            }
        )
    return records


def write_jsonl(path: Path, records: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def write_span_bloom(span_inventory: Dict[str, SpanAggregate], output_path: Path, error_rate: float = 1e-5) -> None:
    max_elements = max(1_000, len(span_inventory) * 2)
    bloom = BloomFilter(max_elements=max_elements, error_rate=error_rate)
    for span_id in span_inventory.keys():
        bloom.add(span_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump({"error_rate": error_rate, "count": len(span_inventory), "bloom": bloom}, handle)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    span_inventory, inventory_path = ensure_span_inventory(args.manifest, args.span_inventory, args.rebuild)
    span_list = list(span_inventory.values())
    context_path = inventory_path.with_name("contexts.jsonl")
    write_context_table(span_inventory, output_path=context_path)
    bloom_path = inventory_path.with_name("spans.bloom")
    write_span_bloom(span_inventory, bloom_path)

    positives = sample_positive(span_list, count=POSITIVE_COUNT, rng=rng)
    dev_claims = read_dev_claims(args.dev_jsonl)
    negatives = sample_negatives(claims=dev_claims, span_ids=set(span_inventory.keys()), count=POSITIVE_COUNT, rng=rng)
    structural = build_structural_twins(span_list, count=STRUCT_COUNT, rng=rng)
    semantic = build_semantic_twins(span_list, count=SEMANTIC_COUNT, rng=rng)
    contexts = build_context_queries(span_list, count=CONTEXT_COUNT, rng=rng)

    output_dir = args.output_dir
    write_jsonl(output_dir / "queries_exact_pos.jsonl", positives)
    write_jsonl(output_dir / "queries_exact_neg.jsonl", negatives)
    write_jsonl(output_dir / "queries_struct_twin.jsonl", structural)
    write_jsonl(output_dir / "queries_sem_twin.jsonl", semantic)
    write_jsonl(output_dir / "queries_contexts.jsonl", contexts)

    print(f"Wrote SBI query sets to {output_dir}")
    print(f"Context table written to {context_path}")
    print(f"Span bloom written to {bloom_path}")


if __name__ == "__main__":
    main()
