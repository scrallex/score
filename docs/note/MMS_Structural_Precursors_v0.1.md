# Structural Precursors in MMS Telemetry via Percentile-Calibrated Manifolds (v0.1)

## Abstract
We transform MMS telemetry into structural bit streams (UP/ACCEL/RANGEEXP/ZPOS) and analyse them with a native STM manifold whose router thresholds are re-tuned per slice to maintain 5–7 % foreground coverage. In the 00:00–01:00 storm hour we recover four high-confidence structural proposals and three storm→storm twins that align 50 windows each against 2017-09-13 (mean ANN distance ≈2×10⁻³). Foreground density rises to 7.4 % in the final five minutes before onset, a trend that persists across adjacent slices while a quiet-day baseline remains flat. All artefacts (states, configs, proposals, twins, diagnostics, plots) are reproducible via one-line CLI tooling.

## 1. Method
1. Bit encode MMS1 FGM channels into structural tokens (`__UP`, `__ACCEL`, `__RANGEEXP`, `__ZPOS`) at 1 s cadence.
2. Run the STM native kernel (QFH/QBSA) with window_bytes=1024, stride=512, capturing coherence, entropy, stability, rupture, and λ per window plus signature q-grams.
3. Apply percentile-based guardrail calibration (`scripts/calibrate_router.py`) that relaxes coherence/entropy/stability percentiles until the slice foreground coverage lies within 5–20 % (observed 5–7 %).
4. Enforce structural-only outputs (`scripts/enforce_structural.py`) so strings, proposals, and twins are restricted to `__` tokens with connector ≥0.5.
5. Index with ANN/postings, generate proposals and storm→storm twins, and summarise diagnostics and lead-time densities via the new CLI shims.

Full reproducibility commands are listed in `docs/note/methods_repro.md`.

## 2. Results
### 2.1 Midnight reference (2017-09-08 00:00–01:00)
- **Figures:** `fig1_mms_0000_overview.png`, `fig2_mms_0000_zoom.png`, `fig3_mms_0000_lead.png`
- **Tables:** `tab1_scorecard.csv` (row “2017-09-08 00:00–01:00”), `tab2_midnight_proposals.csv`, `tab3_midnight_twins.csv`, `tab4_leadtime_bins_midnight.csv`
- **JSON artefacts:** `analysis/mms_0000_state.json`, `analysis/router_config_0000.json`, `analysis/mms_0000_proposals_struct.json`, `analysis/mms_twins_0000_to_0913.json`, `analysis/mms_twins_0000_diagnostics.json`, `analysis/mms_0000_leadtime.json`

Key findings:
- Guardrail keeps foreground at 5.7 % with coherence 7.7×10⁻³, entropy 0.9923, stability 0.465.
- Four z/up proposals survive (ANN distance 0.7–1.4×10⁻³) with connectors = 0.5.
- Three storm→storm twins align 50 windows each against 2017-09-13 (mean ANN 1.97×10⁻³) and share the signature q-gram `c0.01_s0.49_e0.99`.
- Lead-time density climbs from 5.2 %→7.4 % in the final 20→0 min before onset (see also onset sweep in `tab4a_midnight_onset_sweep.csv`).

### 2.2 Neighbour slice (2017-09-08 01:00–02:00)
- **Figures:** `fig4_mms_0100_overview.png`, `docs/plots/mms_0100_lead.png`
- **Tables:** `tab1_scorecard.csv` (row “2017-09-08 01:00–02:00”), `tab4_leadtime_bins_neighbor.csv`
- **JSON:** `analysis/mms_0100_state.json`, `analysis/router_config_0100.json`, `analysis/mms_0100_proposals_struct.json`, `analysis/mms_twins_0100_to_0913.json`, `analysis/mms_twins_0100_diagnostics.json`, `analysis/mms_0100_leadtime.json`

This slice mirrors the midnight vocabulary (four structural proposals, three 50-window twins with mean ANN ≈2.14×10⁻³) and exhibits a similar rise to 7.3 % foreground in the final bin, confirming hour-over-hour stability.

### 2.3 Additional evidence
- 22:30–23:30 retains two proposals and a single 50-window twin (ANN 2.33×10⁻³) with a sharper late-bin surge (13 %); see `analysis/mms_proposals_struct_2017-09-07_2230-2330.json`, `analysis/mms_twins_2230_diagnostics.json`, `analysis/mms_2230_leadtime.json`.
- 23:00–00:00 keeps four proposals but only one twin and a muted lead-time ramp, underscoring why midnight is the reference case (`analysis/mms_2300`* artefacts).
- Quiet baseline (2017-09-10) shows 5.9 % foreground, purely `__UP` proposals, generic twins, and a flat density profile (8.8 % first to last bin); refer to `analysis/mms_quiet_*` JSON and `docs/plots/mms_quiet_*`.

## 3. Sensitivity & Baseline
- **Percentile variants:** P90/20/75 and P88/22/72 return 0 % coverage, validating the need for the guardrail relaxer.
- **Stride change (1 s vs 0.5 s):** Foreground remains 5.7 %; proposals persist though twins drop under strict thresholds (documented in `analysis/mms_0000_stride256_state.json`).
- **Feature ablation:** RangeExp-only and Accel-only seeds preserve structural proposals but fail to reproduce the 50-window twins, indicating joint z/up signatures carry the cross-day signal.
- **Quiet baseline:** See scorecard and diagnostics; no structural precursors appear.
- **Onset sweep:** `tab4a_midnight_onset_sweep.csv` summarises six onset hypotheses between 00:20–00:50 UTC; the final-bin density remains ≥3.6 % (peaking 7.4 %) in every case.

## 4. Limitations & Next Steps
- Absolute coherence remains small (~10⁻²), expected for MMS; onset timing is manual.
- Plan automated onset labelling, multi-channel/joint bit features, and validation across additional missions (THEMIS, Cluster).
- Explore permutation testing for the lead-time ramp and multi-hour sweeps to quantify precursor robustness.

## 5. Artifact Index
All referenced files live under `/docs/note/` (figures, tables, reproducibility sheet) and `/analysis/` (states, configs, proposals, twins, diagnostics, leadtime). The helper CLI entry points `stm-plots` and `stm-leadtime` regenerate plots and lead summaries; `scripts/make_scorecard.py` refreshes the scorecard.
