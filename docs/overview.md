# Sep Text Manifold Overview

The Sep Text Manifold (STM) is a quantum-inspired framework for analyzing text corpora to uncover hidden patterns, repetitive motifs, and thematic structures. Drawing from the SEP Engine's Quantum Field Harmonics (QFH) algorithms—originally developed for financial time-series analysis—STM adapts these techniques to treat text as a byte stream, detecting "echoes" of repetition and stability at the bit level.

## Core Principles

STM processes text without traditional NLP tokenization or semantics, focusing instead on raw byte transitions to compute informational metrics:

1. **Byte-Level Processing**: Text files are read as UTF-8 bytes and concatenated into a continuous stream (with separators). This preserves the natural "rhythm" of the data.

2. **Sliding Windows and Manifolds**: Overlapping windows (default: 2048 bytes, 1024-byte stride) are analyzed to produce a *manifold*—a chronological sequence of *dynamic fingerprints*. Each fingerprint includes:
   - **Metrics**: Coherence (pattern consistency, [0,1]), Stability (persistence, [0,1]), Entropy (uncertainty, [0,1]), Rupture (structural breaks, [0,1]).
   - **Signature**: A coarse hash (e.g., "c0.85_s0.92_e0.15") for repetition detection.
   - **Hazard**: Lambda hazard rate, indicating collapse risk.

3. **String Extraction and Aggregation**: Alphanumeric tokens (words/phrases) are extracted and aligned to covering windows. Metrics are aggregated (means) per string, yielding profiles of how "patternable" (high coherence/stability, low entropy/rupture) each string is across its occurrences.

4. **Graph-Based Themes**: Strings are connected via co-occurrence in windows (filtered by PMI and degree). Community detection identifies *themes*—clusters of related motifs. Graph analytics (betweenness, entropy) highlight *connectors*—strings bridging themes.

5. **Discovery and Querying**: Tools generate "bridge strings" (proposals matching seed centroids) and enable fast retrieval (ANN + q-grams in the *seen* engine) for trigger-based pattern surfacing.

This approach reveals non-semantic structures: repetitive phrases in stable contexts, thematic bridges, and noisy vs. ordered regions—useful for document analysis, motif mining, or anomaly detection.

## Architecture Layers

- **C++ Core (QFH Engine)**: Native bitstream analysis for metrics (qfh.cpp/h, manifold_builder.cpp/h). Bound via pybind11 (sep_quantum.cpp) for Python acceleration. Fallback Python implementations in encode.py.

- **Python Pipeline**: Ingestion (ingest.py), windowing/manifold (manifold.py), extraction/aggregation (strings.py), theming (themes.py), scoring (scoring.py). Full orchestration in pipeline.py.

- **Utilities**: CLI (cli.py) for ingest/summary/propose; API (api.py) for serving; seen.py for indexed queries; binary logging (binary_log.py).

- **Data Flow**:
  ```
  Text Directory → Ingest Bytes → Sliding Windows → QFH Metrics → Manifold Signals
                  ↓
  Extract Tokens → Align to Windows → Aggregate Profiles → Co-occurrence Graph → Themes/Scores
                  ↓
  CLI/API/Seen Queries → Proposals/Themes/Retrievals
  ```

## Usage

Install: `pip install -e .` (requires C++ build for native).

CLI Example:
```
stm ingest ./docs --output state.json  # Analyze directory
stm strings --input state.json --top 10  # Top patternable strings
stm themes --input state.json  # List themes
stm propose --input state.json --seeds "quantum,metrics" --k 5  # Bridge proposals
stm discover --input state.json --mode cross-theme --theme-a 0 --theme-b 2
stm index build --state state.json --postings analysis/signature_postings.json --ann analysis/ann.hnsw --ann-meta analysis/ann.meta
python -m sep_text_manifold.seen   # verifies /stm/seen foreground/deferred router
```

API: Run `uvicorn sep_text_manifold.api:app` for endpoints like `/strings/search`, `/discover`.

For integration with full SEP Engine, see [integration_with_sep.md](integration_with_sep.md).

## Limitations and Extensions

- **Current Scope**: Focuses on English alphanumeric tokens; extend regex in strings.py for phrases/n-grams.
- **Performance**: Native QFH scales to large corpora; Python fallbacks for prototyping.
- **No Semantics**: Bit-level ignores meaning—combine with embeddings for hybrid analysis.
- **Themes**: Relies on co-occurrence; tune graph params (min_pmi, max_degree) in pipeline.py.

STM transforms text into a "quantum manifold" for pattern discovery. Future work: Streaming ingestion, vector embeddings, multi-language support.

Last Updated: 2025-09-19
