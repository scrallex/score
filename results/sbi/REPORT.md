# Span Receipts Index Benchmark Report

<!-- Copy fresh metrics from JSON summaries under `results/sbi/`. Leave TODO markers only while runs are in flight. -->

## A. Exact Membership

| Metric | Value |
| --- | --- |
| Precision | TODO |
| Recall | TODO |
| False Positive Rate | TODO |
| Bloom FPR | TODO |
| Bloom Target FPR | TODO |
| Median TP Latency (ms) | TODO |
| Median TN Latency (ms) | TODO |

- Source: `results/sbi/membership_summary.json`
- Queries: `data/sbi/queries_exact_pos.jsonl`, `data/sbi/queries_exact_neg.jsonl`

## B. Structural Twin Retrieval

| Metric | @1 | @3 | @5 | @10 |
| --- | --- | --- | --- | --- |
| Recall | TODO | TODO | TODO | TODO |
| Precision | TODO | TODO | TODO | TODO |

- p50 latency (ms): TODO
- p90 latency (ms): TODO
- Recall by edit distance: TODO (copy from JSON)
- Source: `results/sbi/struct_summary.json`
- Queries: `data/sbi/queries_struct_twin.jsonl`

## C. Semantic Twin Retrieval

| Metric | Value |
| --- | --- |
| Recall@10 | TODO |
| Mean Reciprocal Rank | TODO |
| p50 latency (ms) | TODO |
| p90 latency (ms) | TODO |

- Source: `results/sbi/sem_summary.json`
- Queries: `data/sbi/queries_sem_twin.jsonl`

## D. SBI Context Ranking

| Metric | Value |
| --- | --- |
| Recall@k (mean) | TODO |
| Mean Reciprocal Rank | TODO |
| nDCG | TODO |
| Reinforcement Spearman ρ | TODO |
| p90 latency (ms) | TODO |

- Source: `results/sbi/contexts_summary.json`
- Queries: `data/sbi/queries_contexts.jsonl`

## E. Gate (/seen + /sbi) Evaluation Checklist

1. Serve gate: `make serve-sri`
2. Run benches against the live manifest: `make sbi-bench`
   - Capture combined throughput/latency from `results/sbi/bench_latest.json`
3. Optional calibration sweep (if a classifier head is attached): document admit/margin thresholds and calibration plots in `results/sbi/calibration/`
4. Next actions: note follow-ups required before release (e.g., regenerate spans after corpus changes, widen query coverage, rotate credentials)

## Artifacts & Environment

- Pack manifest: `analysis/truth_packs/example/manifest.json`
- SBI span inventory: `analysis/truth_packs/example/sbi/spans.jsonl`
- Context table: `analysis/truth_packs/example/sbi/contexts.jsonl`
- Membership bloom: `analysis/truth_packs/example/sbi/spans.bloom`
- Queries: `data/sbi/`
- Benchmarks: `results/sbi/`
- Commit / environment: TODO

## Notes & Follow-ups

- [ ] Compare measured bloom FPR against target (≤ 2× configured error rate).
- [ ] Plot latency histograms for structural/semantic tasks.
- [ ] Confirm gate calibration curves align with reliability expectations (Brier, ECE) if a classifier head is enabled.
