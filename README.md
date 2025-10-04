# Sep Text Manifold (STM)

STM is a coherence engine: it builds structural manifolds over any corpus and
checks generated answers against them so every line carries its receipts. By
fusing structural rhythm (patternability/coherence/hazard) with semantic
intent, STM blocks hallucinated spans, repairs them from nearest twins, and
returns citations for what survives.

**Quick win:** `make semantic-guardrail-demo`

This command rebuilds the documentation truth-pack, runs the evaluation set, and
opens the **Caseboard**: raw LLM answer on the left, per-sentence decision log in
the middle (repeat/hazard/semantic badges with twin citations), and the repaired
answer on the right with numbered receipts. The top rail shows hallucination
rate, repair yield, citation coverage, and latency percentiles for the run.

Head to `docs/INDEX.md` for a guided tour of whitepapers, runbooks, design
notes, and evaluation assets.

### Reality Filter Toolkit

- Build a truth-pack manifold: `make pack PACK=docs_demo`
- Stream spans (sim or LLM) through the reality filter: `make stream PACK=docs_demo -- SPANS=demo/truth_pack/sample_spans.json`
- Benchmark the `/seen` service: `PYTHONPATH=src .venv/bin/python scripts/benchmark_seen.py --manifest analysis/truth_packs/docs_demo/manifest.json`
- Run the FastAPI shim locally: `UVLOOP=1 gunicorn scripts.reality_filter_service:app -k uvicorn.workers.UvicornWorker -w 2 --bind 0.0.0.0:8000 --keep-alive 5`
- Generate sweeps/permutation/report artefacts: `make sweep PACK=docs_demo`, `make permutation PACK=docs_demo`, `make report PACK=docs_demo`

## Structural Intelligence Demo (mxbikes.xyz)

- Generate the canned payload from MMS artefacts: `make demo-payload`.
- Run the FastAPI backend locally: `python -m stm_demo --reload` (defaults to
  `http://127.0.0.1:8000`).
- Serve the static site in `webapp/` or run the docker stack: `make demo-up`.

The demo site pulls from `/api/demo`, renders the Pattern Prophet, Twin Finder
and Context Refinery summaries, and ships with an nginx config wired for
`mxbikes.xyz` TLS certificates.

This repository is intended as a **framework** – it contains skeleton
modules, command line entry points and API scaffolding.  You can run
basic ingestion and analysis out of the box, but the heavy lifting (the
QFH/QBSA algorithms, manifold build logic and advanced scoring) should
be ported or wrapped from the existing SEP repositories.  See
`docs/integration_with_sep.md` for details on where to pull the core
implementation from.

## Quick start

To use this project you will need Python 3.10 or higher.  Create a
virtual environment and install the package in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
# Build the optional native QFH/QBSA bindings (requires a C++20 toolchain and TBB)
pip install -e .[native]
# Install the optional Transformer-based reliability model (requires PyTorch)
pip install -e .[attn]
```

Once installed you can ingest a corpus and perform an initial analysis
using the command line interface:

```bash
stm ingest /path/to/text/data --window-bytes 2048 --stride 1024 --summary-top 15
stm summary
stm strings --top 50
stm themes
stm propose --seeds signature,manifold_builder --min-connector 0.35 --top 10
stm discover --theme-a 0 --theme-b 2 --min-connector 0.3 --top 10

# Build routing indices (signature q-grams + ANN)
stm index build \
  --state analysis/score_state_native.json \
  --postings analysis/signature_postings.json \
  --ann analysis/ann.hnsw --ann-meta analysis/ann.meta

# Verify foreground/deferred routing
PYTHONPATH=src python - <<'PY'
from sep_text_manifold.seen import get_engine
print(get_engine().seen('allocator'))
PY

# Follow the manifold log to keep the router hot
PYTHONPATH=src python - <<'PY'
from sep_text_manifold.stream import follow_log
from sep_text_manifold.seen import get_engine

engine = get_engine()
for record in follow_log('analysis/manifold.log'):
    engine.update_window(record)
    break  # remove to stream continuously
PY
```

For details on the commands and available options run `stm --help`.

### Training the reliability Transformer

If you install the `attn` extra, start by converting a public fact-verification
corpus (FEVER or SciFact) into STM's evaluation format:

```bash
# FEVER example (needs the wiki snapshot alongside the JSONL split)
python scripts/convert_fever_to_eval.py \
  data/fever/train.jsonl results/eval/fever_train/eval_detail.jsonl \
  --wiki-pages data/fever/wiki-pages --split train --progress
```

Then train the O-space Transformer reliability head.  The harness now supports
calibration sweeps, richer evidence encodings, and attention-entropy
regularisation:

```bash
PYTHONPATH=src python scripts/train_reliability_attn.py \
  results/eval/fever_train/eval_detail.jsonl \
  --epochs 5 --batch-size 32 --device cuda \
  --calibrate-thresholds --admit-threshold 0.5 --margin-threshold 0.25 \
  --output-checkpoint models/reliability_fever.pt
```

The trainer logs admit precision/recall, macro-F1, Brier score, Expected
Calibration Error, and attention entropy for the validation split.  Use
`--dry-run` to execute a single forward pass and `--disable-phase-channel` or
`--disable-cross-attention` when running ablations.  The saved checkpoint packs
the model weights, configuration (including evidence encoder settings), and
tokeniser vocabulary so it can be loaded via
`reality_filter_eval.py --reliability-model`.

During evaluation the reliability wrapper now feeds full evidence sentences and
their structural metrics into the Transformer; each repaired span records the
model's admit probability and support margin in the `reliability_trace` field of
`eval_detail.jsonl`, making calibration diagnostics easy to audit.

## Reproducing the PlanBench++ and CodeTrace experiments

The repository now ships with end-to-end harnesses for the planning and coding
benchmarks discussed in the STM whitepaper.

### PlanBench++ guardrail pipeline

```
make planbench-all
```
### Logistics guardrail demo

Run the logistics disruption scenario, optional twin lookup, and figure generation with the commands below.

```bash
# Base trace + metrics (writes analysis/, tokens/, timeline.json, dashboard.html)
PYTHONPATH=score/src python score/scripts/logistics_guardrail_demo.py \
  --output-root analysis/logistics_guardrail_demo

# Re-run with cached twins using the freshly written STM state
PYTHONPATH=score/src python score/scripts/logistics_guardrail_demo.py \
  --output-root analysis/logistics_guardrail_demo_with_twins \
  --twin-state analysis/logistics_guardrail_demo/analysis_state.json \
  --twin-top-k 2 --twin-max-distance 0.4

# Export the whitepaper figures referenced in docs/whitepaper/logistics_guardrail.md
PYTHONPATH=score/src python score/scripts/plot_logistics_guardrail_figures.py \
  --timeline analysis/logistics_guardrail_demo_with_twins/timeline.json \
  --output-dir docs/whitepaper/img
```

The guardrail applies calibrated lambda and dilution thresholds (see `analysis/router_config_logistics_invalid_native.json`). In the reference run the first alert fires at step 5 with lambda 0.538 and the classical validator fails at step 8, yielding a three-step lead. Passing `--twin-state` enables `sep_text_manifold.suggest_twin_action`, which surfaces recovery precedents in the dashboard's **Recovery Recommendation** panel. Permutation evidence in `docs/tests/permutation_logistics_native.json` shows precision 1.0 with foreground coverage close to two percent.

Validate the pipeline at any time with `pytest score/tests/test_logistics_guardrail_demo.py -q`, and consult `docs/whitepaper/logistics_guardrail.md` for narrative context and figure references.



This target regenerates the synthetic PlanBench dataset (default 300 instances
per domain), converts traces into STM manifolds, calibrates the 5% guardrail for
each domain, runs 20k-shuffle permutation tests, and writes guardrail sweep
results to `docs/note/appendix_guardrail_sweep.csv`.

To explore lower guardrails or tighten statistical reporting, run the sweep
script directly. The example below scans 1–8% in 0.5% increments, emits per
target router configs, writes permutation summaries, and stores a structured
JSON artifact with coverage, lead, and p-value confidence intervals:

```
python scripts/guardrail_sweep.py \
  analysis/blocksworld_invalid_state.json \
  output/planbench_by_domain/blocksworld \
  --grid 0.01 0.08 0.005 \
  --prefix blocksworld_low_guardrail \
  --summary-json analysis/blocksworld_guardrail_sweep.json \
  --label Blocksworld

# Drop the Logistics guardrail to 2.5% automatically when the 5% config
# fails permutation significance (writes both base and dynamic artefacts).
python scripts/calibrate_router.py \
  output/planbench_by_domain/logistics/invalid_state.json \
  --target-low 0.05 --target-high 0.07 \
  --domain-root output/planbench_by_domain/logistics \
  --dynamic-target 0.025 --dynamic-window 0.005 \
  --pvalue-threshold 0.05 --pvalue-metric min \
  --use-native-quantum \
  --output analysis/router_config_logistics_invalid_5pct.json

When the native bindings are installed, add `--use-native-quantum` to
`scripts/planbench_to_stm.py`, `scripts/enrich_features.py`, and
`scripts/experiments/build_causal_domain.py` to ensure the manifolds are built
with the QFH/QBSA kernel rather than the simplified fallback.

If VAL traces are not available you can synthesise them directly from the
generated plans:

```
.venv/bin/python scripts/generate_synthetic_traces.py --root data/planbench_public --overwrite
```

Use `--optimize-permutation` when you want calibration to search nearby
coverage targets and retain the guardrail with the strongest permutation
significance. The helper flags `--optimize-width`, `--optimize-span`,
`--optimize-samples`, and `--optimize-centers` control the search window, letting
you bias the scan toward bespoke coverage budgets or previously audited
thresholds without writing additional scripts.

# Enrich Blocksworld or Mystery twins with the Logistics corpus and rerun sweeps.
# (The make target does this automatically via --enrich-from when rebuilding domains.)
python scripts/enrich_twin_corpus.py \
  output/planbench_by_domain/blocksworld/gold_state.json \
  --extra output/planbench_by_domain/logistics/gold_state.json \
  --note "logistics-twin-enrichment"

To add further corpora (for example bug-fix commits or scaled PlanBench
exports) set `PLANBENCH_EXTRA_TWINS` when running the make target. Each
entry should point to a `gold_state.json` that you want merged into the
Blocksworld and Mystery foreground libraries:

```
PLANBENCH_EXTRA_TWINS="data/twins/bugfix_state.json data/twins/robotics_state.json" \
  make planbench-all

# Example using built-in corpora:
PLANBENCH_EXTRA_TWINS="output/planbench_public/gold_state.json analysis/mms_state.json" \
  make planbench-all
```
```

### CodeTrace comparison suite

```
make codetrace-report
```

This command replays the CodeTrace maintenance tasks and emits the aggregated
report used in the whitepaper.

The `CodeTraceAdapter` now inspects Python edits for structural cues, parsing
added snippets into AST categories (functions, branches, context managers,
etc.) and recording change magnitudes. These features feed directly into the
manifold, giving the guardrail additional leverage in discriminating noisy
maintenance edits from meaningful repairs.

## Repository structure

- `src/sep_text_manifold/` – the Python package containing ingestion,
  encoding, manifold construction, the analysis pipeline, string
  scoring, theme detection and API definitions.
- `docs/` – design notes and integration guidelines.
- `tests/` – pytest-based regression suite covering the pipeline and
  CLI.
- `cpp/` – optional pybind11-ready scaffolding for the native hot loop.

## Contributing

This skeleton is deliberately minimal.  It is designed to get you to a
point where you can start experimenting with the SEP methodology on
text data.  Contributions are welcome!  Please open issues or pull
requests with enhancements, bug fixes or documentation improvements.
