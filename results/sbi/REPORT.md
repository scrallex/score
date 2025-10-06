# Span Receipts Index Benchmark Report

<!-- Fill the tables below by copying values from the JSON summaries under `results/sbi/`. -->

## A. Exact Membership

| Metric | Value |
| --- | --- |
| Precision | |
| Recall | |
| False Positive Rate | |
| Bloom FPR | |
| Bloom Target FPR | |
| Median TP Latency (µs) | |
| Median TN Latency (µs) | |

- Source: `results/sbi/membership_summary.json`
- Queries: `data/sbi/queries_exact_pos.jsonl`, `data/sbi/queries_exact_neg.jsonl`

## B. Structural Twin Retrieval

| Metric | @1 | @3 | @5 | @10 |
| --- | --- | --- | --- | --- |
| Recall | | | | |
| Precision | | | | |

- p50 latency (ms):
- p90 latency (ms):
- Recall by edit distance: (≤1) __ | (≤2) __ | (≤3) __
- Source: `results/sbi/struct_summary.json`
- Queries: `data/sbi/queries_struct_twin.jsonl`

## C. Semantic Twin Retrieval

| Metric | Value |
| --- | --- |
| Recall@10 | |
| MRR | |
| p50 latency (ms) | |
| p90 latency (ms) | |

- Source: `results/sbi/sem_summary.json`
- Queries: `data/sbi/queries_sem_twin.jsonl`

## D. SBI Context Ranking

| Metric | Value |
| --- | --- |
| Recall@k (mean) | |
| MRR | |
| nDCG | |
| Reinforcement Spearman ρ | |
| p90 latency (ms) | |

- Source: `results/sbi/contexts_summary.json`
- Queries: `data/sbi/queries_contexts.jsonl`

## E. Gate (/seen) Evaluation Checklist

1. Serve gate: `PYTHONPATH=src uvicorn scripts.reality_filter_service:app --host 0.0.0.0 --port 8000`
2. Benchmark latency: `PYTHONPATH=src python scripts/benchmark_seen.py --manifest analysis/truth_packs/fever_train_full_final/manifest.json --concurrency 16 --duration 60 --out results/sbi/seen_latency.json`
3. Precision/recall sweep: `PYTHONPATH=src python scripts/reality_filter_eval.py --pack analysis/truth_packs/fever_train_full_final/manifest.json --reliability-model results/models/fever_reliability.pt --calibrate-thresholds --val-ratio 0.10 --progress --out results/sbi/gate_calibration.json`
4. Target: Precision ≥ 0.90 at Recall ≥ 0.30, receipts coverage ≥ 0.90, p90 latency ≤ 50 ms, throughput ≥ target.

## Artifacts & Environment

- Pack manifest: `analysis/truth_packs/fever_train_full_final/manifest.json`
- SBI span inventory: `analysis/truth_packs/fever_train_full_final/sbi/spans.jsonl`
- Context table: `analysis/truth_packs/fever_train_full_final/sbi/contexts.jsonl`
- Membership bloom: `analysis/truth_packs/fever_train_full_final/sbi/spans.bloom`
- Queries: `data/sbi/`
- Benchmarks: `results/sbi/`
- Commit / environment: *fill in before release*

## Notes & Follow-ups

- [ ] Compare measured bloom FPR against target (≤ 2× configured error rate).
- [ ] Plot latency histograms for structural/semantic tasks.
- [ ] Cross-check gate calibration curves align with reliability expectations (Brier, ECE).
