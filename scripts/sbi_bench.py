#!/usr/bin/env python3
"""Benchmark suite for the Span Receipts Index experiments."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr

from sbi.index import SpanReceiptsIndex
from sbi.span_corpus import compute_span_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--pack",
            type=Path,
            default=Path("analysis/truth_packs/example/manifest.json"),
            help="Path to the truth-pack manifest used for lookups.",
        )
        p.add_argument("--out", type=Path, required=True)

    membership = subparsers.add_parser("membership", help="Benchmark exact membership lookups.")
    add_common(membership)
    membership.add_argument("--queries", nargs="+", type=Path, required=True, help="Positive then negative query JSONL paths.")

    structural = subparsers.add_parser("structural", help="Benchmark structural twin retrieval.")
    add_common(structural)
    structural.add_argument("--queries", type=Path, required=True)
    structural.add_argument("--k", type=int, default=10)

    semantic = subparsers.add_parser("semantic", help="Benchmark semantic twin retrieval.")
    add_common(semantic)
    semantic.add_argument("--queries", type=Path, required=True)
    semantic.add_argument("--k", type=int, default=10)

    contexts = subparsers.add_parser("contexts", help="Benchmark SBI context ranking.")
    add_common(contexts)
    contexts.add_argument("--queries", type=Path, required=True)
    contexts.add_argument("--k", type=int, default=10)

    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


def describe_latencies(times_ns: Sequence[int]) -> Dict[str, float]:
    if not times_ns:
        return {"p50_ms": 0.0, "p90_ms": 0.0, "mean_ms": 0.0}
    arr = np.array(times_ns, dtype=np.float64) / 1e6
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
    }


def membership_bench(args: argparse.Namespace) -> Dict[str, object]:
    index = SpanReceiptsIndex(args.pack)
    positives_path = args.queries[0]
    negatives_paths = args.queries[1:]
    positives = load_jsonl(positives_path)
    negatives: List[Dict[str, object]] = []
    for path in negatives_paths:
        negatives.extend(load_jsonl(path))

    tp = fp = tn = fn = 0
    bloom_fp = bloom_fn = 0
    bloom_tp = bloom_tn = 0
    tp_times: List[int] = []
    tn_times: List[int] = []

    for record in positives:
        span_id = record.get("span_id") or compute_span_id(record.get("text", ""))
        start = time.perf_counter_ns()
        bloom_hit = index.bloom_contains(span_id)
        actual_hit = index.contains(span_id)
        elapsed = time.perf_counter_ns() - start
        if actual_hit:
            tp += 1
            tp_times.append(elapsed)
        else:
            fn += 1
        if bloom_hit:
            bloom_tp += 1
        else:
            bloom_fn += 1

    for record in negatives:
        span_id = record.get("span_id") or compute_span_id(record.get("text", ""))
        start = time.perf_counter_ns()
        bloom_hit = index.bloom_contains(span_id)
        actual_hit = index.contains(span_id)
        elapsed = time.perf_counter_ns() - start
        if actual_hit:
            fp += 1
            tp_times.append(elapsed)
        else:
            tn += 1
            tn_times.append(elapsed)
        if bloom_hit:
            if not actual_hit:
                bloom_fp += 1
        else:
            bloom_tn += 1

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    fpr = fp / max(1, tn + fp)
    bloom_fpr = bloom_fp / max(1, len(negatives))
    bloom_fn_rate = bloom_fn / max(1, len(positives))

    metrics = {
        "positives": len(positives),
        "negatives": len(negatives),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fpr,
        "lookup_latency_tp": describe_latencies(tp_times),
        "lookup_latency_tn": describe_latencies(tn_times),
        "bloom": {
            "error_rate_target": index.bloom_meta.get("error_rate", 0.0),
            "false_positive_rate": bloom_fpr,
            "false_negative_rate": bloom_fn_rate,
            "tp": bloom_tp,
            "tn": bloom_tn,
            "fp": bloom_fp,
            "fn": bloom_fn,
        },
    }
    return metrics


def structural_bench(args: argparse.Namespace) -> Dict[str, object]:
    index = SpanReceiptsIndex(args.pack)
    queries = load_jsonl(args.queries)
    k = args.k
    ks = [1, 3, 5, k]
    ks = sorted(set(min(k, x) for x in ks))
    hit_counts = {key: 0 for key in ks}
    precision_sums = {key: 0.0 for key in ks}
    latency_ns: List[int] = []
    per_distance: Dict[int, Dict[str, float]] = {}
    per_distance_counts: Dict[int, int] = {}

    for record in queries:
        query_text = record["query"]
        target_id = record["span_id"]
        edit_distance = int(record.get("edit_distance", 0))
        start = time.perf_counter_ns()
        matches = index.structural_search(query_text, top_k=k)
        elapsed = time.perf_counter_ns() - start
        latency_ns.append(elapsed)
        predicted_ids = [m.span_id for m in matches]
        for key in ks:
            top_ids = predicted_ids[:key]
            hit = 1 if target_id in top_ids else 0
            hit_counts[key] += hit
            precision_sums[key] += hit / key
        bucket = per_distance.setdefault(edit_distance, {"hits": 0.0, "total": 0.0})
        bucket["total"] += 1.0
        if target_id in predicted_ids[:k]:
            bucket["hits"] += 1.0
        per_distance_counts[edit_distance] = per_distance_counts.get(edit_distance, 0) + 1

    results = {
        "query_count": len(queries),
        "latency": describe_latencies(latency_ns),
        "recall": {str(key): hit_counts[key] / max(1, len(queries)) for key in ks},
        "precision": {str(key): precision_sums[key] / max(1, len(queries)) for key in ks},
        "recall_by_edit_distance": {
            str(dist): (bucket["hits"] / max(1.0, bucket["total"])) for dist, bucket in per_distance.items()
        },
    }
    return results


def semantic_bench(args: argparse.Namespace) -> Dict[str, object]:
    index = SpanReceiptsIndex(args.pack)
    queries = load_jsonl(args.queries)
    k = args.k
    ks = sorted({1, 3, 5, k})
    hit_counts = {key: 0 for key in ks}
    rr_sum = 0.0
    latency_ns: List[int] = []

    for record in queries:
        query_text = record["query"]
        targets = {item["span_id"] for item in record.get("targets", [])}
        start = time.perf_counter_ns()
        matches = index.semantic_search(query_text, top_k=k)
        elapsed = time.perf_counter_ns() - start
        latency_ns.append(elapsed)
        predicted_ids = [m.span_id for m in matches]
        first_rank = None
        for idx, span_id in enumerate(predicted_ids, start=1):
            if span_id in targets:
                first_rank = idx
                break
        if first_rank is not None:
            rr_sum += 1.0 / first_rank
        for key in ks:
            if any(span_id in targets for span_id in predicted_ids[:key]):
                hit_counts[key] += 1

    results = {
        "query_count": len(queries),
        "latency": describe_latencies(latency_ns),
        "recall": {str(key): hit_counts[key] / max(1, len(queries)) for key in ks},
        "mrr": rr_sum / max(1, len(queries)),
    }
    return results


def ndcg_score(relevances: Sequence[int]) -> float:
    if not relevances:
        return 0.0
    gains = [(2 ** rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(relevances)]
    dcg = float(sum(gains))
    ideal = sorted(relevances, reverse=True)
    idcg = float(sum((2 ** rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(ideal)))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def contexts_bench(args: argparse.Namespace) -> Dict[str, object]:
    index = SpanReceiptsIndex(args.pack)
    queries = load_jsonl(args.queries)
    k = args.k
    recall_sum = 0.0
    mrr_sum = 0.0
    ndcg_sum = 0.0
    latency_ns: List[int] = []
    reinforcements: List[float] = []
    correctness: List[int] = []

    for record in queries:
        span_id = record["span_id"]
        gt_left = {ctx["text"] for ctx in record.get("left", [])}
        gt_right = {ctx["text"] for ctx in record.get("right", [])}
        gt_all = gt_left | gt_right
        start = time.perf_counter_ns()
        preds_left = index.contexts(span_id, "left", top_k=k)
        preds_right = index.contexts(span_id, "right", top_k=k)
        elapsed = time.perf_counter_ns() - start
        latency_ns.append(elapsed)
        predictions = preds_left + preds_right
        if not predictions or not gt_all:
            continue
        hits = 0
        first_rank = None
        relevances: List[int] = []
        for idx, pred in enumerate(predictions, start=1):
            text = pred["text"]
            score = pred.get("reinforcement", 0.0)
            is_hit = 1 if text in gt_all else 0
            hits += is_hit
            relevances.append(is_hit)
            reinforcements.append(score)
            correctness.append(is_hit)
            if is_hit and first_rank is None:
                first_rank = idx
        recall_sum += min(1.0, hits / max(1, len(gt_all)))
        if first_rank is not None:
            mrr_sum += 1.0 / first_rank
        ndcg_sum += ndcg_score(relevances)

    if reinforcements and len(set(reinforcements)) > 1 and len(set(correctness)) > 1:
        corr, _ = spearmanr(reinforcements, correctness)
        correlation = float(corr)
    else:
        correlation = 0.0

    count = len(queries)
    results = {
        "query_count": count,
        "latency": describe_latencies(latency_ns),
        "recall_mean": recall_sum / max(1, count),
        "mrr": mrr_sum / max(1, count),
        "ndcg": ndcg_sum / max(1, count),
        "reinforcement_correlation": correlation,
    }
    return results


def main() -> None:
    args = parse_args()
    if args.command == "membership":
        payload = membership_bench(args)
    elif args.command == "structural":
        payload = structural_bench(args)
    elif args.command == "semantic":
        payload = semantic_bench(args)
    else:
        payload = contexts_bench(args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
