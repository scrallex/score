# Reality Filter Methods Note (Draft)

## 1. Motivation

Workslop and hallucinated spans erode trust in AI copilots. We introduce a reality filter that requires every generated span to repeat a trusted structural signature under low hazard and to carry citations (twins).

## 2. Structural Admission Rule

- Echo requirement: repetitions ≥ `r_min` within the target manifold.
- Hazard gate: structural hazard λ ≤ `λ_max`.
- Semantic alignment: cosine similarity/margin ≥ `σ_min`.
- Decisions with `/seen` service payload: `{repeat_ok, hazard_ok, semantic_ok, admit}`.

## 3. Target Manifold & Twins

- Truth-pack ingestion (`scripts/reality_filter_pack.py`).
- ANN + q-gram search for twin retrieval.
- Acceptance curves collected via sweep target (`make sweep`).

## 4. Caseboard Demo

- Span sources: simulated JSON (`SimSpanSource`), evaluation claims (`data/eval/<pack>/claims.jsonl`), and LLM streaming (`LLMSpanSource`).
- Evaluation runner (`scripts/reality_filter_eval.py`) executes baseline vs filter, logging per-sentence decisions, repairs, and latency.
- Caseboard (`scripts/demos/reality_filter_caseboard.py`) shows raw answer, decision log (repeat/hazard/semantic badges with twin citations), and repaired answer with numbered receipts.

## 5. Results Snapshot

| Pack | Hallucination ↓ | Repair Yield ↑ | Citation Coverage ↑ | Latency p50 (ms) | Latency p90 (ms) |
| --- | --- | --- | --- | --- | --- |
| docs_demo | 1.00 | 1.00 | 0.00 | 85 | 85 |
| whitepaper_demo | 0.40 | 0.50 | 1.00 | 85 | 89 |

Commands:

```bash
make pack PACK=docs_demo
PYTHONPATH=src .venv/bin/python scripts/make_eval_from_pack.py \
  --manifest analysis/truth_packs/docs_demo/manifest.json \
  --output data/eval/docs_demo/claims.jsonl
PYTHONPATH=src .venv/bin/python scripts/reality_filter_eval.py \
  --manifest analysis/truth_packs/docs_demo/manifest.json \
  --claims data/eval/docs_demo/claims.jsonl \
  --pack-id docs_demo
PYTHONPATH=src .venv/bin/python scripts/reality_filter_report.py --packs docs_demo
```

Equivalent commands apply to `whitepaper_demo`.

## 6. Risks & Limits

- Span-level validation (not full document proofs).
- Geometry sensitivity (window bytes, corpus bias).
- Latency trade-offs (hash vs transformer embeddings).

## 7. Reproducibility

```bash
make pack PACK=<pack>
PYTHONPATH=src .venv/bin/python scripts/make_eval_from_pack.py --manifest analysis/truth_packs/<pack>/manifest.json --output data/eval/<pack>/claims.jsonl
make eval PACK=analysis/truth_packs/<pack> CLAIMS=data/eval/<pack>/claims.jsonl
make eval-report PACK=<pack>
make sweep PACK=<pack>
make permutation PACK=<pack>
make report PACK=<pack>
uvicorn scripts.reality_filter_service:app --reload
make bench-seen PACK_PATH=analysis/truth_packs/<pack>
```

Reports: `results/report/<pack>.md`, evaluation detail: `results/eval/<pack>/eval_detail.jsonl`.
