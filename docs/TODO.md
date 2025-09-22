Paper Scope & Structure

Purpose: Position Paper 2 as an empirical study linking QFH structural signals (coherence, entropy, λ hazard) to symbolic plan quality across PlanBench domains. Clarify that Paper 1 covered instrumentation and MIT PDDL-INSTRUCT targeted LLM accuracy; this work probes whether low-hazard echoes predict plan validity/difficulty.
Background bullets: briefly recap QFH manifold (refer to Paper 1), PDDL-INSTRUCT headline accuracies (BW 94 %, Logistics 79 %, Mystery 64 %), and why PlanBench + CodeTrace form a common evaluation bed.
Contributions list: unified native-metric PlanBench corpus with validity labels; correlation analyses against human/LLM success; twin-filter/λ sweeps; discussion on structural recurrence as a hardness signal.
Data Expansion Tasks

PlanBench (native build ✅):
output/planbench_by_domain/blocksworld_native, .../mystery_bw_native already regenerated. Extract per-window QFH features + guardrail stats (coverage 0.065 @ lead 2.8, mystery coverage 0.058 @ lead 2.25, min p≈0.33/0.115).
Stash aggregated states (blocksworld_state_native.json, mystery_bw_state_native.json) and produce CSVs with success labels if available.
CodeTrace corpus: replayed 3 demo tasks (fix_flaky_test, rename_service_endpoint, resolve_missing_import) → native dataset at output/codetrace_native/. Need to enlarge corpus (more tasks/runs) because current state has 1 window ⇒ guardrail calibration saturates. Plan: gather additional CodeTrace traces (from repo or MIT release) using demo/coding/run_replay.py --variant stm --persist, rerun ingestion script.
Plan validity labels: align each trace/window with success/failure (use MIT reported rates or dataset metadata). If only plan-level labels exist, propagate that label to all windows in the trace.
LLM outputs: locate or request PDDL-INSTRUCT run logs; if accessible, sync them under data/pddl_instruct/ and record reasoning chain lengths, VAL results, etc., for correlation.
Experimental Design Roadmap

Correlations: per domain compute Pearson/Spearman between STM irreversibility (or λ) and plan validity (0/1), plus between QFH metrics and MIT accuracy stats. Produce scatter plots akin to Paper 1’s bridge figure.
Twin-filter sweeps: bucket by predicate/action type or trace length, rerun permutation tests; log buckets where p < 0.05.
λ sweeps: vary hazard weightings (entropy-only, coherence-only, 50/50) and thresholds (0.10–0.20). Plot coverage vs plan success for each blend.
Indicator comparisons: compute traditional difficulty proxies (plan length, branching factor, heuristic cost) and compare with λ/echo metrics to check complementarity.
Ablations: drop coherence/entropy/λ in turn; observe impact on correlations for at least one domain.
Writing & Presentation

Results section template: per domain table (coverage, lead, p-value, mean λ); correlation matrices; histograms of metrics; λ vs success scatter; twin-filter heatmaps.
Discussion focus: highlight patterns (e.g. high λ ↔ failures, domains sensitive to thresholds). Stress limitations (dataset size, p-values still high, λ tuning). Note interactions with PDDL-INSTRUCT (e.g. structural hazard can flag “hard” instances even when LLM succeeds).
Conclusion: emphasise structural manifold as hardness indicator, not predictive alpha; propose feeding λ into LLM feedback, expanding to other symbolic tasks.
Repro Appendix: add script block describing CodeTrace ingestion (run_replay.py --persist, native ingestion script), correlation/twin-filter commands, env vars (VALKEY_URL, HOTBAND_PAIRS, ECHO_*, PDDL_INSTRUCT paths).
Project Management

Create Makefile/CLI target (e.g. make planbench-native-suite) chaining: PlanBench regenerate → CodeTrace ingest → correlations → λ sweeps → figure generation.
Track subtasks (dataset expansion, correlations, sweeps, writing) in issue tracker/Linear.
Document new env requirements in README (paths for CodeTrace/LLM data, hazard sweep parameters).
Plan follow-up: once CodeTrace corpus expanded, re-run guardrail calibration to obtain meaningful coverage/p-values; prepare baseline comparisons against LLM outputs.
Next actionable steps: 1) source additional CodeTrace traces (from MIT dataset or internal runs), rerun the native ingestion script; 2) gather PlanBench plan-validity labels (success/failure) and integrate into aggregated states to enable correlations.