# Semantic Augmentation for STM

This experiment shows how to bridge the byte-level STM manifold with a
semantic embedding view so AI-facing applications can reason about
meaning without abandoning the structural telemetry that the system
already provides.

## Workflow

1. Ingest any corpus with `stm ingest --store-signals` to produce a
   state file (e.g. `analysis/score_state_native.json`).
2. Run the bridge demo, pointing it at semantic seed phrases that
   represent the intent you care about:

```bash
PYTHONPATH=src scripts/semantic_bridge_demo.py \
  analysis/score_state_native.json \
  --seeds "risk" "volatility" "rupture" \
  --top-k 8 \
  --min-occurrences 3 \
  --embedding-method auto \
  --output results/semantic_bridge.json
```

3. Inspect the JSON output. Three sections are emitted:
   - `top_structural` – strings the manifold already ranks highly by
     patternability.
   - `top_semantic` – strings closest to your seed phrases in the
     embedding space.
   - `top_combined` – strings that balance both views.

4. Compare the Pearson correlation reported in `statistics` against
   your expectations. A low correlation highlights cases where semantic
   intent surfaces new candidates the structural manifold missed, while
   a high correlation means the existing guardrails already capture the
   theme.

## Embedding Backends

- When `sentence-transformers` is installed the script loads the
  specified model (defaults to `all-MiniLM-L6-v2`).
- Without the dependency it falls back to a deterministic hash-based
  embedding so the demo still runs in restricted environments.

## Extending the Demo

- Feed the combined rankings into the existing `/stm/seen` router to
  prime semantic triggers.
- Use the `semantic_similarity` scores as additional features in guardrail calibration.
- Swap the seed list to cover narratives, code intents, or telemetry anomalies and compare how structural vs semantic alerts differ.

The script lives at `scripts/semantic_bridge_demo.py` and uses the
helpers in `src/sep_text_manifold/semantic.py`.

