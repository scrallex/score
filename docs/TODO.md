Here’s a structured roadmap to build on your foundation paper and to set up your next round of research.  The aim is to turn the current methods paper into a springboard for deeper empirical results and stronger STM↔spt integration.

## 1. Consolidate and check the existing pipeline

* **Verify the end-to-end workflow**. Now that the native metrics flow is stable, run through the reproducibility commands in the appendix from start to finish (PlanBench ingestion, feature enrichment, guardrail calibration, synthetic generation, Valkey priming, snapshot export, figure generation) to ensure there are no hidden gaps. This will also confirm that the restored cooldown/hysteresis helpers don’t interfere with the live loop.

* **Document environment assumptions**. Make sure colleagues know to set `VALKEY_URL`, `HOTBAND_PAIRS`, `ECHO_*` parameters, and (optionally) OANDA keys when priming live data. Clear environment defaults reduce friction when others reproduce your work.

## 2. Expand the datasets

* **Broaden PlanBench coverage**. Logistics is only one domain; regenerate Blocksworld, CodeTrace and Mystery traces with `--use-native-quantum`. Calibrate guardrails and compute coverage/lead/p‑values for each domain. This will test whether the native QFH metrics generalise across textual environments.

* **Add more FX instruments and asset classes**. Thirty‑day snapshots for eight majors gave you hazard clusters around 0.12–0.18. Extend to crosses (EUR/GBP, JPY/CHF), metals (XAU/USD), and possibly crypto if feeds are available. Also lengthen the sample (90 or 180 days) to see if hazard regimes persist. Use the same prime/export scripts to collect snapshots and update the λ‑calibration curves.

* **Gather higher‑frequency synthetic corpora**. The current synthetic benchmark uses only a few canonical patterns. Generate larger synthetic bitstreams with variable burst lengths, multi‑level alternations and regime shifts to stress the event model and refine your intuition about when coherence/entropy and λ diverge.

## 3. Improve statistical power

* **Twin‑filtering experiments**. On PlanBench traces, bucket windows by action distribution (e.g. predicate classes) or trace length, then rerun permutation tests within each bucket. The idea is to reduce heterogeneity and see whether p‑values drop when the manifold is evaluated on more homogeneous slices.

* **Lambda weighting and thresholds**. The hazard λ is currently a blend of entropy and coherence; experiment with alternative weightings (entropy‑only, coherence‑only, or dynamic weights that adapt with volatility) to see how admission curves change. Similarly, test different λ thresholds (0.10, 0.15, 0.20) on live FX data to quantify trade‑off between coverage and noise.

* **Correlate with classical indicators**. Compute standard technical indicators (ATR, RSI, MACD) over the same FX snapshots and compare them to λ and repetition counts. See whether high echo/low hazard windows correspond to particular indicator regimes, and whether the manifold offers orthogonal information.

## 4. Deepen the STM↔spt bridge

* **Richer contingency tables**. In the bridge analysis you used a single hazard threshold (0.25) and found 62 % agreement between STM irreversibility and market rupture. Produce a full sensitivity analysis: vary the hazard threshold and STM irreversibility cutoff and record agreement/disagreement rates. This will help you tune thresholds for real‑time use.

* **Temporal alignment**. Align PlanBench windows and FX snapshots by calendar date and time of day. Investigate whether certain time‑of‑day patterns show stronger irreversibility–rupture correlation (e.g. during London/New York overlap).

* **Feature translation**. Use the native kernel to generate QFH metrics directly from PlanBench bitstreams (beyond logistic features) and compare them to spt metrics. This may reduce the need for STM proxies and reveal deeper structural similarities.

## 5. Start planning the next paper(s)

* **PlanBench/STP performance paper**. Use the expanded datasets and twin‑filtering results to write a second paper focused on significance. Aim to show whether native metrics plus filtering strategies reduce p‑values below 0.05 or improve lead times relative to baselines.

* **Combined STM↔spt paper**. Once you have more FX pairs and a richer bridge analysis, draft a third paper that evaluates the manifold as a trading filter. Compare gated vs. ungated strategies on historical data; include profit/risk metrics and benchmark against classical indicators. Make sure to emphasise that the goal is stability and regime detection rather than pure alpha.

* **Methodological spin‑offs**. Consider papers on hazard calibration methods, the mathematical properties of the QFH event process, or an application to other domains (e.g. sensor data, social streams) to broaden the technology’s appeal.

## 6. Continue improving the infrastructure

* **Automate CI and data regeneration**. Integrate the `--use-native-quantum` paths and the figure generation scripts into a nightly job that refreshes synthetic, PlanBench and FX datasets. This ensures your experiments always run on up‑to‑date data and helps catch regressions early.

* **Refine the Python/C++ interface**. If you plan to release the package publicly, consider adding type hints, docstrings and examples for the native APIs, and align the naming conventions between STM and spt code paths to reduce friction for external contributors.

By following this roadmap you’ll systematically deepen the statistical validity of your findings, broaden the data base, and prepare compelling follow‑up papers.  The current foundation paper gives you a solid platform; the next phase should focus on data diversity, rigorous significance tests, and comparative performance against standard baselines.
