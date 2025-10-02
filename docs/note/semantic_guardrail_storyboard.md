# Semantic Guardrail Storyboard

## Purpose

Define the narrative arc and technical flow for the "Semantic Guardrail"
demo and whitepaper chapter that leverages the structural/semantic scatter
plots generated in this repository.

## Core Claim

The QFH/STM manifold provides a structural score that is independent of
meaning. Sentence-transformer embeddings provide semantic alignment that is
independent of structure. Plotting these two axes together reveals three
regions:

- **Semantic but unstable** (upper left): conceptually relevant events that
  are noisy in the raw telemetry/logs.
- **Structural but irrelevant** (lower right): highly repetitive operations
  that do not match the intent (e.g., API boilerplate, steady CRON jobs).
- **Hybrid high-confidence pocket** (upper right): recurring, orderly events
  whose semantic content matches our risk/resilience seeds. This sparse
  quadrant is the actionable signal that neither pure time-series nor pure NLP
  systems can isolate.

## Demo Components

1. **Event stream simulator**
   - Generates interleaved events drawn from an annotated library (see
     `results/semantic_bridge_docs.json` and
     `results/semantic_bridge_mms.json`).
   - Emits structure/semantic scores from pre-computed manifolds to mimic
     live QFH + embedding evaluation.
   - Supports scripted "incident" sequences (e.g., repeated
     `database_connection_timeout`).

2. **Dual baseline panels**
   - *Naïve semantic guardrail*: triggers when semantic similarity to the seed
     set exceeds a threshold, regardless of structural stability.
   - *SEP structural guardrail*: triggers when structural patternability
     exceeds a threshold, regardless of semantics.

3. **Hybrid scatter view**
   - Real-time scatter plot (reuse `scripts/semantic_bridge_plot.py` logic) with
     highlighted points when an event crosses both thresholds.
   - Displays per-point tooltips (`string`, `semantic`, `patternability`,
     `coherence`, `occurrences`).

4. **Alert compositor**
   - Emits textual alerts only when an event hits the hybrid region. Alerts
     include recommended action, derived from a configurable mapping.
   - Optional webhook/log sink to drive dashboards.

## Data Preparation

- **Corpus A (Documentation/API)**: `analysis/semantic_demo_state.json` with
  enriched scatter artifacts under `results/semantic_bridge_docs.json`.
- **Corpus B (Telemetry/MMS)**: `analysis/mms_state.json` with scatter artifacts
  in `results/semantic_bridge_mms.json`.
- Future corpus: real-time market or infrastructure feeds (pending).

## Implementation Tasks

1. Build `scripts/semantic_guardrail_stream.py` that:
   - Loads one or more state files and their semantic projections.
   - Streams synthetic events to stdout/JSONL with structural and semantic
     scores.
   - Supports deterministic scenarios (baseline + high-confidence incident).

2. Extend `scripts/semantic_bridge_plot.py` to offer a "live" mode where new
   events update the scatter instead of generating only static PNGs.

3. Author documentation section (whitepaper + README snippet) explaining the
   two-axis interpretation using the combined plot.

4. Package demo runbook:
   - `make semantic-guardrail-demo` target that
     1. Regenerates the manifolds (optional),
     2. Produces semantic JSON/plots if missing,
     3. Launches the stream simulator, and
     4. Spins up a lightweight dashboard (Matplotlib/Plotly or textual).

## Cross-Industry Pitch Points

- **SRE/DevOps**: Stability of infrastructure signals + semantic alignment to
  incident vocabulary.
- **Finance/Risk**: Structural repetition in trade/market feeds + semantic
  ties to risk/volatility terminology.
- **Manufacturing/IoT**: Machine telemetry motifs + semantics around failure,
  maintenance, anomaly.

Each pitch references the hybrid quadrant as the differentiator over
traditional anomaly detectors or keyword scanners.

