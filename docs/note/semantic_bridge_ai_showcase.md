# Semantic + Structural Manifold Showcase for AI Workflows

## Objective

Demonstrate how the structural manifold can pair byte-level repetition with
semantic intent so AI copilots and guardrails understand both *how often* a
pattern appears and *what it means*.

## Dataset

- Corpus: `/score/docs` filtered to Markdown, text and JSON files (45 files,
  ~0.87 MB).
- STM ingest command:

  ```bash
  PYTHONPATH=/score .venv/bin/stm ingest docs \
    --extensions md txt json yaml \
    --output analysis/semantic_demo_state.json \
    --store-signals --min-token-len 3 --drop-numeric --verbose
  ```

- Result: 853 manifold windows, 5,595 scored strings, 1,607 structural themes.

## Semantic Bridge Setup

- SentenceTransformer: `all-MiniLM-L6-v2` (CPU wheel).
- Seeds chosen to cover reliability-heavy operations work:
  `risk`, `resilience`, `volatility`, `anomaly`, `predictive maintenance`.
- Outputs:
  - JSON alignment report: `results/semantic_bridge_docs.json`.
  - Scatter plot fusing patternability and semantic similarity:
    `results/semantic_bridge_scatter.png`.
  - Markdown digest: `results/semantic_bridge_report.md`.
  - Stream simulator output: `results/semantic_guardrail_stream.jsonl` (see
    `scripts/semantic_guardrail_stream.py`).

## Key Findings

- **High-signal overlap**: strings such as `failures`, `disruption`, `loss`, and
  `robust` climb to the top because they are both structurally repetitive in the
  docs *and* semantically close to the reliability seeds.
- **Structural blind spots**: purely structural ranking surfaces API plumbing
  tokens (`ping`, `demo_id`). Combining semantics demotes them, making room for
  industry-relevant cues that matter to resilience teams.
- **Thematic spread**: annotated scatter plot highlights clusters—finance terms
  (`loss`, `volatility`) share the frontier with logistics/operations
  vocabulary (`maintain`, `disruption`). This is the hook for cross-industry
  storytelling: the manifold detects repeated failure narratives no matter the
  domain, while embeddings map them to the AI user's intent.
- **Lead features for AI copilots**:
  - Use `semantic_similarity` as a guardrail feature when triaging plan traces
    or code diffs.
  - Feed top combined strings into `/stm/seen` to pre-warm semantic triggers
    for incident response bots.
  - Blend the scatter output into dashboards so stakeholders can see which
    structural motifs align with risk-focused language.

## Next Experiments

1. **Industry overlays** – Run the bridge against logistics traces, software
   repos, and financial market snapshots to publish side-by-side scatter plots
   (stacked via `results/semantic_bridge_combined.png`).
2. **Temporal semantics** – Stream new manifolds through
   `scripts/semantic_bridge_demo.py` in “follow-the-log” mode to monitor when
   semantic risk terms flare up despite stable structural metrics.
3. **Guardrail infusion** – Extend `scripts/calibrate_router.py` to accept a
   semantic prior, lowering coverage thresholds when semantic similarity spikes
   around high-rupture motifs.

The artefacts above show that the manifold already captures the *rhythm* of the
data; sentence-level embeddings layer in meaning so AI systems can respond with
context-aware recommendations across industries. The stream simulator ties the
static plot to operator experience by contrasting naïve semantic alerts,
structure-only alerts, and the hybrid guardrail that fires only on the
upper-right quadrant.
