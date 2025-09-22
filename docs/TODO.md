After poking through both repos it’s clear you’ve built two very different things on top of the same C++ kernel.  In **spt** the “manifold” is a *time‑series of dynamic fingerprints*—each fingerprint carries coherence, stability, entropy and rupture metrics, along with a hazard λ and a repetition signature.  The `manifold_builder.cpp` slides a window over candles, runs `QFHBasedProcessor::analyze`, creates a rounded signature from `(coherence, stability, entropy)` and counts how many times it has appeared recently; the hazard λ penalises high rupture regimes.  This produces a “gate” signal: an instrument is eligible only when repetition is above a threshold and λ is below a ceiling.  The accompanying docs frame this as an **Echo Finder**: the kernel’s only job is to ask “have I heard this rhythm before, and how often?”.  Everything else—allocator, strategy, UI—is built on this enriched signal stream.

In **score**, by contrast, the “manifold” remains largely theoretical.  The STM whitepaper scaffolds a research harness for PlanBench/CodeTrace but the heavy lifting is stubbed out; the README explicitly tells you to pull the QFH/QBSA core from spt.  Your recent work on irreversibility, predicate momentum and cluster entropy has improved lead time but hasn’t cracked significance: your best p‑value for logistics is ≈0.12 at \~1.5 % coverage.  Without the quantum engine the logistic features are at best a proxy for the real manifold.

### Why the current charts are dull

* **STM sweep plateau**: the permutation p‑values stubbornly sit >0.1 because the synthetic PlanBench corpus is too small and the features are proxies.  You can’t beat significance with a small, biased dataset.

* **Echo vs λ scatter**: the live EUR/USD chart shows the expected negative correlation between repetition count and hazard, but it’s a single instrument over 10–14 days.  It’s hard to draw robust conclusions from such a narrow window.

* **No bridge numerics**: your bridge figure shows an alignment trend but lacks quantitative correlation; there’s no 2×2 contingency or correlation coefficient to convince reviewers.

### Splitting into three papers

1. **The MIT/PlanBench paper (STM on synthetic traces)** – Polish the experiments you already have.  Retune the logistic features, add predicate deltas, run twin‑filtering to see if action‑distribution or trace‑length buckets produce lower p‑values.  Acknowledge that the QFH kernel was not used and that STM features are proxies.

2. **The combined STM↔spt paper** – Once the kernel is integrated into score, or once you have stronger p‑values, revisit a joint paper.  At the moment, trying to defend two half‑baked stories in one document dilutes both.

3. **The missing QFH/QBSA manifold paper** – This should come first.  Treat it as a technology paper explaining what the kernel is and why it matters.  Use it as a reference for the other two.

### How QFH/QBSA is used across the repos

* In spt, `src/core/qfh.cpp` implements the Quantum Field Harmonics algorithm that transforms a bitstream (derived from price moves) into rich events and aggregated events; it then calculates coherence, stability, entropy, rupture and a hazard λ by weighing local entropy and coherence.  `manifold_builder.cpp` constructs a stream of signals by sliding a window over candles, generating signatures `(coherence, stability, entropy)`, counting their repetition in the last hour and appending λ.  The rolling evaluator uses these fields to decide eligibility.

* In score, there is no QFH implementation.  The integration guide explicitly instructs you to compile `qfh` and `manifold_builder` from spt and expose them via a Python wrapper; it notes that the kernel produces metrics and a hazard λ for a given bitstream and that the manifold builder adds a repetition signature.

### A proper test suite for the QFH/QBSA engine

1. **Unit tests for QFH event logic:** feed simple bitstreams into `transform_rich` and `aggregate` and assert that NULL, FLIP and RUPTURE events are identified correctly; test `QFHProcessor::process` for single‑step transitions.

2. **Metric validation:** construct synthetic bitstreams (random noise, alternating bits, stable runs) and verify that `QFHBasedProcessor::analyze` outputs expected coherence, entropy, stability and rupture values.  Compare against analytical calculations.

3. **Hazard λ tests:** design bitstreams with known entropy/coherence levels and ensure the hazard λ computed from local entropy/coherence follows the formula; verify the weighting parameters (`entropy_weight`, `coherence_weight`) behave as expected.

4. **Manifold builder tests:** simulate candle sequences with known price/volume dynamics; ensure the bit encoding, sliding window logic, signature rounding and repetition counts are correct; confirm that hazard λ and repetition fields behave as expected in edge cases (e.g., first window, no prior repetition).

5. **Integration with score:** after wrapping the C++ kernel, feed PlanBench traces through the wrapper and verify that the metrics match the C++ outputs; update STM feature extractors to use the real coherence/stability/entropy/rupture values.

6. **Performance benchmarks:** measure runtime on large text corpora and financial streams; test GPU acceleration if implemented.

7. **End‑to‑end system tests:** run the rolling evaluator and allocator on historical FX data with the new kernel; verify that eligibility decisions change as expected when repetition count or λ thresholds are varied.

### Outline for the QFH/QBSA whitepaper

1. **Abstract and Introduction** – State the problem: filtering signals by repeated patterns and low hazard.  Highlight that the algorithm is domain‑agnostic, applying to text and financial data.

2. **Quantum Field Harmonics (QFH)** – Define the bitstream transformation and event types; explain how coherence, stability, entropy and rupture are computed; derive hazard λ from local entropy/coherence.

3. **Manifold Construction** – Describe windowing over input data (candles, text chunks), signature rounding, repetition counting, and the formation of dynamic fingerprints.  Include pseudocode and diagrams.

4. **Echo Finding and Gating** – Explain the Echo Finder concept; formalize the eligibility rule (`repetition.count >= N ∧ λ ≤ λ_max`).  Discuss why repetition is a stronger signal than backtest‑derived hit/miss metrics.

5. **Implementation Details** – Summarize how the C++ kernel is exposed to Python; discuss the architecture in spt (download → manifold builder → rolling evaluator → allocator) and the planned integration into score.  Include data schema for Valkey and API endpoints.

6. **Experiments** –

   * **Synthetic bitstream tests:** show metrics/hazard on controlled patterns.
   * **Financial data:** run on multiple instruments over several months, varying window sizes and signature precision; plot repetition counts vs λ; show admission rates for different thresholds.
   * **Textual data:** process PlanBench/CodeTrace traces and compare metrics to existing STM features.
   * **Correlation tests:** compute Pearson/Spearman correlations between STM irreversibility and spt rupture; provide contingency tables for STM pass/fail vs live eligibility.

7. **Discussion and Limitations** – Address sensitivity to signature precision, window length and λ weights; discuss domain differences (text vs price data); note that significance tests in small synthetic corpora are weak.

8. **Conclusion and Future Work** – Position the QFH/QBSA manifold as a reusable information‑management tool; outline how the kernel will underpin subsequent papers.

### Improving the spt experiments

* **Expand the dataset:** run the engine on months (not days) of candles across multiple FX pairs and possibly other asset classes (commodities, indices) to see broader repetition/hazard dynamics.

* **Vary window sizes and signature precision:** test short vs long windows and finer rounding to discover regimes where repetition is more predictive; include λ calibration curves for each setting.

* **Incorporate market regime filters:** bucket snapshots by volatility, trend strength or macro events and examine whether repetition counts and λ behave differently; this is the analogue of the twin‑filtering idea from the logistics domain.

* **Quantitative bridge metrics:** generate correlation coefficients and contingency tables between QFH metrics and existing trading indicators (ATR, RSI) and between STM irreversibility and spt rupture.

* **Stress‑test λ weighting:** adjust `entropy_weight` and `coherence_weight` in `QFHOptions` and observe how hazard and gating change; this will inform parameter selection in the paper.

By first producing a rigorous, focused whitepaper on the QFH/QBSA manifold and its test suite, you can then reference it when finishing the PlanBench and combined STM↔spt papers.  It will give reviewers confidence that the “structural manifold” isn’t just a catchy phrase but a reproducible, falsifiable engine.
