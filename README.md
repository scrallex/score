# Sep Text Manifold (STM)

The **Sep Text Manifold (STM)** project is a starting point for building a
system that analyses large volumes of text data using the quantum‑inspired
analysis developed in the SEP Engine.  It takes directories of arbitrary
textual files, translates them into a stream of bits, and computes
informational metrics (coherence, stability, entropy and rupture) over
sliding windows of the data.  These metrics are then aggregated back
onto the words and phrases that appear in the text, allowing you to
identify high‑quality patterns, connectors between topics and overall
themes in the corpus.

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

## Reproducing the PlanBench++ and CodeTrace experiments

The repository now ships with end-to-end harnesses for the planning and coding
benchmarks discussed in the STM whitepaper.

### PlanBench++ guardrail pipeline

```
make planbench-all
```

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
  --output analysis/router_config_logistics_invalid_5pct.json

# Enrich Blocksworld or Mystery twins with the Logistics corpus and rerun sweeps.
# (The make target does this automatically via --enrich-from when rebuilding domains.)
python scripts/enrich_twin_corpus.py \
  output/planbench_by_domain/blocksworld/gold_state.json \
  --extra output/planbench_by_domain/logistics/gold_state.json \
  --note "logistics twin enrichment"

To add further corpora (for example bug-fix commits or scaled PlanBench
exports) set `PLANBENCH_EXTRA_TWINS` when running the make target. Each
entry should point to a `gold_state.json` that you want merged into the
Blocksworld and Mystery foreground libraries:

```
PLANBENCH_EXTRA_TWINS="data/twins/bugfix_state.json data/twins/robotics_state.json" \
  make planbench-all
```
```

### CodeTrace comparison suite

```
make codetrace-report
```

This command replays the CodeTrace maintenance tasks and emits the aggregated
report used in the whitepaper.

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
