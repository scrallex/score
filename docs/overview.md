# Overview

The Sep Text Manifold project is inspired by the SEP Engine’s ability
to uncover hidden order in live data streams by analysing transitions
in a stream of bits.  While the original SEP focus was on financial
market data, the underlying theory is domain‑agnostic.  Any data
source that can be reduced to a sequence of bytes can be analysed in
the same way.  This project adapts the SEP methodology to the problem
of **bulk text analysis**.

Key ideas:

1. **Byte‑level encoding** – All text is treated as raw bytes.  We do
   not tokenize or normalise into words before computing the quantum
   metrics.  This is important because the SEP algorithms operate on
   transitions between bits, not on higher‑level semantics.

2. **Sliding windows** – We process the byte stream using overlapping
   windows (e.g., 2 KB windows with a 1 KB stride).  For each window we
   compute quantum metrics such as coherence, stability, entropy and
   rupture.  These metrics summarise how structured or noisy the
   recent data is.

3. **Manifold build** – The metrics for each window are assembled
   chronologically into a **manifold**.  A manifold is essentially
   an ordered list of “dynamic fingerprints” summarising the recent
   information flow, along with repetition signatures and hazard
   estimates.

4. **String extraction** – After computing signals on the byte level,
   we tokenise the original text into words and phrases.  Each string
   occurrence is aligned back onto the windows that overlap it.
   Aggregating the window metrics over all occurrences of a string
   yields measures of how coherent, stable, entropic or ruptured that
   string tends to be.  This allows us to surface strings that have
   unusually high patternability or act as connectors between
   different topics.

5. **Theme detection** – Strings are connected via co‑occurrence
   graphs.  Community detection over this graph yields clusters of
   strings representing themes in the corpus.  Additional graph
   analytics identify connector strings that bridge themes.

This repository now includes an executable pipeline
(`sep_text_manifold.pipeline.analyse_directory`) and a CLI (`stm`) that
run these steps end-to-end on a directory of text files.  The pipeline
produces manifold windows, aggregated string profiles, connector
scores, theme assignments, bridge-string proposals (`stm propose`,
`stm discover`) and can be surfaced through a lightweight demo API.
The actual quantum
algorithms are not reimplemented here; instead, you should port or
wrap the QFH and QBSA implementations from the core SEP repositories
(see `integration_with_sep.md`).
