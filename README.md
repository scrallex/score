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

# Build the routing indices (signature postings + ANN)
PYTHONPATH=src python - <<'PY'
import json
from collections import defaultdict
from pathlib import Path
state = json.load(open('analysis/score_state_native.json'))
signals = sorted(state['signals'], key=lambda s: s['id'])
postings = defaultdict(set)
q=3
for i in range(len(signals)-q+1):
    qgram = tuple(signals[j]['signature'] for j in range(i, i+q))
    postings[qgram].add(signals[i+q-1]['id'])
Path('analysis').mkdir(exist_ok=True)
json.dump({"\u001f".join(k): sorted(v) for k,v in postings.items()}, open('analysis/signature_postings.json','w'))
PY

PYTHONPATH=src python - <<'PY'
import json, numpy as np, hnswlib
state=json.load(open('analysis/score_state_native.json'))
vecs=[[s['metrics']['coherence'], s['metrics']['stability'],
       s['metrics']['entropy'], s['metrics']['rupture'], s['lambda_hazard']] for s in state['signals']]
ids=[s['id'] for s in state['signals']]
vecs=np.asarray(vecs, dtype=np.float32); ids=np.asarray(ids, dtype=np.int64)
index=hnswlib.Index(space='l2',dim=vecs.shape[1])
index.init_index(max_elements=len(vecs), ef_construction=200, M=32)
index.add_items(vecs, ids); index.set_ef(100)
index.save_index('analysis/ann.hnsw')
json.dump({'dim': vecs.shape[1], 'count': len(vecs)}, open('analysis/ann.meta','w'))
PY

# Call the context router once (foreground vs deferred)
PYTHONPATH=src python - <<'PY'
from sep_text_manifold.seen import get_engine
print(get_engine().seen('allocator'))
PY
```

For details on the commands and available options run `stm --help`.

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
