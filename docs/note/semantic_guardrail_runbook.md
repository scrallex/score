# Semantic Guardrail Demo Runbook

## Prerequisites

- Install the project in editable mode with native extras and embeddings:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -e .
  pip install "sentence-transformers" "matplotlib" "pandas" "tabulate" "hnswlib"
  ```
- Generate the documentation corpus manifold (20–25s on a laptop):
  ```bash
  PYTHONPATH=/score .venv/bin/stm ingest docs \
    --extensions md txt json yaml \
    --output analysis/semantic_demo_state.json \
    --store-signals --min-token-len 3 --drop-numeric
  ```
- (Optional) Generate the MMS telemetry manifold:
  ```bash
  PYTHONPATH=/score .venv/bin/stm ingest data/mms \
    --output analysis/mms_state.json --store-signals
  ```

## Build Semantic Projections

```bash
PYTHONPATH=src .venv/bin/python scripts/semantic_bridge_demo.py \
  analysis/semantic_demo_state.json \
  --seeds risk resilience volatility anomaly "predictive maintenance" \
  --top-k 15 --min-occurrences 3 \
  --embedding-method transformer \
  --output results/semantic_bridge_docs.json

PYTHONPATH=src .venv/bin/python scripts/semantic_bridge_plot.py \
  analysis/semantic_demo_state.json \
  --seeds risk resilience volatility anomaly "predictive maintenance" \
  --embedding-method transformer \
  --output results/semantic_bridge_scatter.png
```

Repeat the commands for `analysis/mms_state.json` to generate telemetry
artifacts (`results/semantic_bridge_mms.json`,
`results/semantic_bridge_mms_scatter.png`). Combine plots:

```bash
python - <<'PY'
from PIL import Image
left = Image.open('results/semantic_bridge_scatter.png')
right = Image.open('results/semantic_bridge_mms_scatter.png')
canvas = Image.new('RGB', (left.width + right.width, max(left.height, right.height)), 'white')
canvas.paste(left, (0, 0))
canvas.paste(right, (left.width, 0))
canvas.save('results/semantic_bridge_combined.png')
PY
```

## Warm the Router (optional for `/stm/seen` demo)

```bash
PYTHONPATH=/score .venv/bin/stm index build \
  --state analysis/semantic_demo_state.json \
  --postings analysis/semantic_demo_postings.json \
  --ann analysis/semantic_demo_ann.hnsw \
  --ann-meta analysis/semantic_demo_ann.meta

mkdir -p analysis/semantic_router
cp analysis/semantic_demo_state.json analysis/semantic_router/score_state_native.json
cp analysis/router_config.json analysis/semantic_router/router_config.json
cp analysis/semantic_demo_postings.json analysis/semantic_router/signature_postings.json
cp analysis/semantic_demo_ann.hnsw analysis/semantic_router/ann.hnsw
cp analysis/semantic_demo_ann.meta analysis/semantic_router/ann.meta
```

Query the router with semantic triggers:

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
from sep_text_manifold.seen import SeenEngine
engine = SeenEngine(base_path='analysis/semantic_router')
for trigger in ['failures', 'disruption', 'loss', 'robust']:
    res = engine.seen(trigger)
    print(trigger, 'foreground', len(res['foreground']))
PY
```

## Stream Simulation (Semantic Guardrail)

```bash
PYTHONPATH=src .venv/bin/python scripts/semantic_guardrail_stream.py \
  --seeds risk resilience volatility anomaly "predictive maintenance" \
  --samples 6
```

Review `results/semantic_guardrail_stream.jsonl` to compare naïve semantic
alerts, structure-only alerts, and the hybrid guardrail alert.

## Integration Hooks

- Feed the JSONL stream into a dashboard or plotly animation to show the
  scatter filling in over time.
- Map `hybrid_guardrail_alert` events to escalation workflows or Slack/Teams
  webhooks.
- Adjust `--incident-*` parameters to demonstrate different severity levels.

