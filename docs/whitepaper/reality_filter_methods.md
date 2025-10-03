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

## 4. Reality Filter Demo

- Span sources: simulated JSON (`SimSpanSource`) and live LLM (`LLMSpanSource`).
- Streaming pipeline (`scripts/reality_filter_stream.py`): evaluation, twin repair loop, telemetry.
- Dashboard panels + KPIs.

## 5. Results (to populate)

- KPIs for docs_demo + second pack.
- Threshold sweeps (coverage vs r_min/λ/σ).
- Permutation p-values.
- Repair yield and citation coverage.
- Latency distribution vs budget.

## 6. Risks & Limits

- Span-level validation (not full document proofs).
- Geometry sensitivity (window bytes, corpus bias).
- Latency trade-offs (hash vs transformer embeddings).

## 7. Reproducibility

- `make pack PACK=<name>`
- `make stream PACK=<name> SPANS=...`
- `make sweep`, `make permutation`, `make report`
- `/seen` service: `uvicorn scripts.reality_filter_service:app --reload`
- Benchmark: `PYTHONPATH=src .venv/bin/python scripts/benchmark_seen.py --manifest ...`

Future revisions will pull figures from `results/report/<pack>.md` and embed structured tables.
