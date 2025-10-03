# Reality Filter Demo Runbook

## Overview

This guide walks through the 10–12 minute "Every line must carry its receipts" demo:

1. Prepare a truth-pack.
2. Run the reality filter stream (sim or LLM source).
3. Launch the dashboard and narrate the three panels + KPIs.
4. Capture artefacts (metrics JSON, sweep CSV, permutation summary).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[native]
pip install httpx uvicorn fastapi
```

## 1. Build a Truth-Pack

```bash
make pack PACK=docs_demo SEEDS="risk resilience volatility anomaly predictive maintenance"
```

Outputs under `analysis/truth_packs/docs_demo/`:

- `manifold_state.json`
- `signature_postings.json`
- `ann.hnsw` / `ann.meta`
- `semantic_scatter.png`
- `manifest.json`

## 2. Stream Spans

Simulated spans:

```bash
make stream PACK=docs_demo SPANS=demo/truth_pack/sample_spans.json
```

LLM spans (requires `OPENAI_API_KEY`):

```bash
PYTHONPATH=src .venv/bin/python scripts/reality_filter_stream.py \
  --manifest analysis/truth_packs/docs_demo/manifest.json \
  --spans demo/truth_pack/sample_spans.json \
  --source llm --repair --r-min 1 \
  --output results/docs_demo_stream.jsonl \
  --metrics-output results/docs_demo_metrics.json
```

## 3. Launch Dashboard

```bash
PYTHONPATH=src .venv/bin/python scripts/demos/semantic_guardrail_dashboard.py \
  --stream results/docs_demo_stream.jsonl \
  --background analysis/truth_packs/docs_demo/semantic_scatter.png \
  --states analysis/truth_packs/docs_demo/manifold_state.json \
  --seeds risk resilience volatility anomaly "predictive maintenance"
```

Story beats:

- **Left panel**: naïve semantic (show hallucinated span).
- **Middle**: naïve structural (high patternability / bad semantics).
- **Right**: hybrid with twins, citations, latency. Highlight repair event and KPI bar.

## 4. Artefacts & Benchmarks

```bash
make sweep PACK=docs_demo
make permutation PACK=docs_demo
make report PACK=docs_demo
PYTHONPATH=src .venv/bin/python scripts/benchmark_seen.py --manifest analysis/truth_packs/docs_demo/manifest.json
```

Collect:

- `results/docs_demo_metrics.json`
- `results/sweeps/docs_demo.csv`
- `results/permutation/docs_demo.json`
- `results/report/docs_demo.md`
- Benchmark output (throughput + latency)

## Notes

- `make clean-demo` removes generated artefacts.
- Use `PACK=<name>` to run the same flow on additional corpora.
- `/seen` service: `uvicorn scripts.reality_filter_service:app --reload`
- Benchmark target: ≥1k spans/sec when using hash embeddings.
