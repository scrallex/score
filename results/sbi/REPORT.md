# Span Receipts Index Benchmark Report

<!-- Fill the tables below by copying values from the JSON summaries under `results/sbi/`. -->

## A. Exact Membership

| Metric | Value |
| --- | --- |
| Precision | 1.0000 |
| Recall | 1.0000 |
| False Positive Rate | 0 |
| Bloom FPR | 0 |
| Bloom Target FPR | 1.0e-05 |
| Median TP Latency (µs) | 25.3 |
| Median TN Latency (µs) | 17.0 |

- Source: `results/sbi/membership_summary.json`
- Queries: `data/sbi/queries_exact_pos.jsonl`, `data/sbi/queries_exact_neg.jsonl`

## B. Structural Twin Retrieval

| Metric | @1 | @3 | @5 | @10 |
| --- | --- | --- | --- | --- |
| Recall | 0.9994 | 1.0000 | 1.0000 | 1.0000 |
| Precision | 0.9994 | 0.3333 | 0.2000 | 0.1000 |

- p50 latency (ms): 2.10
- p90 latency (ms): 5.85
- Recall by edit distance: (≤1) 1.0 | (≤2) 1.0 | (≤3) 1.0
- Source: `results/sbi/struct_summary.json`
- Queries: `data/sbi/queries_struct_twin.jsonl`

## C. Semantic Twin Retrieval

| Metric | Value |
| --- | --- |
| Recall@10 | 0.9999 |
| MRR | 0.9932 |
| p50 latency (ms) | 8.23 |
| p90 latency (ms) | 8.88 |

- Source: `results/sbi/sem_summary.json`
- Queries: `data/sbi/queries_sem_twin.jsonl`

## D. SBI Context Ranking

| Metric | Value |
| --- | --- |
| Recall@k (mean) | 1.0 |
| MRR | 1.0 |
| nDCG | 1.0 |
| Reinforcement Spearman ρ | 0.0 |
| p90 latency (ms) | 0.0034 |

- Source: `results/sbi/contexts_summary.json`
- Queries: `data/sbi/queries_contexts.jsonl`

## E. Gate (/seen) Evaluation Checklist

1. Serve gate *(running in background PID 842094)*: `PYTHONPATH=src uvicorn scripts.reality_filter_service:app --host 0.0.0.0 --port 8000 --log-level warning`
2. Latency/throughput (2000 requests @ 16 concurrent): `PYTHONPATH=src python scripts/benchmark_seen.py --manifest analysis/truth_packs/fever_train_full_final/manifest.json --requests 2000 --concurrency 16 --port 8000 --url http://127.0.0.1:8000 --timeout 30 > results/sbi/seen_latency.json`
   - Result: 1202.7 rps, p50 12.1 ms, p90 14.1 ms, 0 errors (`results/sbi/seen_latency.json`).
3. Reliability sweep (sample of 3000 FEVER dev claims): `PYTHONPATH=src python scripts/reality_filter_eval.py --manifest analysis/truth_packs/fever_train_full_final/manifest.json --claims data/sbi/claims_dev_sample.jsonl --output-dir results/sbi/gate_eval --dev-ratio 0.8 --reliability-model results/models/fever_reliability.pt --reliability-device cpu`
   - Best thresholds: r_min=1, λ≤0.12, σ_min=0.15; macro-F1 (test) 0.17, citation coverage 0 (no admits on this conservative pass).
4. Next actions: expand evaluation to full FEVER dev once compute budget allows; tune reliability gate to reach Precision ≥0.90 at Recall ≥0.30 with ≥90% receipts coverage and p90 latency ≤50 ms.

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
