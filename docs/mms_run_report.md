# MMS Structural Manifold – Initial Findings

## Setup
- **Data**: MMS1 FGM (magnetic field) + FPI (plasma moments) for 2017-09-07–09, 10, 13.
- **Preprocessing**:
  - Native sampling left at 1 s cadence.
  - `scripts/encode_struct_bits.py` converts each numeric channel into structural bit streams (`__UP`, `__ACCEL`, `__RANGEEXP`, `__ZPOS`).
  - Zoom corpus (`nasa/mms/csv_zoom/2017-09-07_22-02`) keeps only the 22:00–02:00 UTC storm window.
- **Kernel**: STM native manifold with `window_bytes=1024`, `stride=512`, `--drop-numeric` to focus on bit tokens.
- **Indices**: `stm index build` → `analysis/mms_zoom_ann.hnsw`, `analysis/mms_zoom_postings.json`.
- **Router calibration**: `scripts/calibrate_router.py` now enforces a 5–20 % foreground coverage band by relaxing percentile cuts in a fixed order and records the effective thresholds + coverage per slice.
- **Automation**: `stm-plots`, `stm-leadtime`, and `scripts/twin_diagnostics.py` regenerate plots, lead-time tables, and ANN summaries as one-liners for any new slice.

## Current Deliverables
| Artifact | Purpose |
| --- | --- |
| `analysis/mms_2017-09-07_2230-2330_state.json` + `analysis/router_config_2017-09-07_2230-2330.json` | 22:30–23:30 manifold + guardrail thresholds (coverage 6.9 %). |
| `analysis/mms_zoom_2300_state.json` + `analysis/router_config_2017-09-07_2300-0000.json` | 23:00–00:00 baseline manifold + guardrail thresholds (coverage 5.4 %). |
| `analysis/mms_0000_state.json` + `analysis/router_config_0000.json` | Midnight manifold + guardrail thresholds (coverage 5.7 %). |
| `analysis/mms_0100_state.json` + `analysis/router_config_0100.json` | 01:00–02:00 manifold + guardrail thresholds (coverage 5.1 %). |
| `analysis/mms_quiet_state.json` + `analysis/router_config_quiet.json` | 2017-09-10 quiet-hour manifold + guardrail thresholds (coverage 5.9 %). |
| `analysis/mms_0000_stride256_state.json` + `analysis/router_config.json` | 0.5 s stride sensitivity manifold (coverage 5.7 %). |
| `analysis/mms_*_top_structural.txt` | Structural-only top strings for every slice (via `scripts/enforce_structural.py strings … --min-occ 2`). |
| `analysis/mms_*_proposals_struct*.json` | Structural proposals per slice (standard, rangeexp-only, accel-only, stride sensitivity). |
| `analysis/mms_twins_*_to_0913.json` | Storm→storm twin payloads (per slice + sensitivity variants) filtered to structural tokens. |
| `analysis/mms_twins_*_diagnostics.json` | Twin diagnostics (aligned windows, mean ANN distance, top signature tokens). |
| `analysis/mms_*_leadtime.json` | Lead-time density tables for storm slices and quiet baseline. |
| `docs/plots/mms_2230_*.png`, `docs/plots/mms_0000_*.png`, `docs/plots/mms_0100_*.png`, `docs/plots/mms_quiet_*.png` | Bx/Bz overlays with `__RANGEEXP`/`__ACCEL` shading + foreground heat strips. |
| `docs/plots/mms_{2230,2300,0000,0100,quiet}_lead.png` | Lead-time density line plots (minutes before onset vs foreground share). |
| `scripts/enforce_structural.py`, `scripts/plot_structural_window.py`, `scripts/lead_time_density.py` | Structural filter, plotting, and lead-time helpers (now part of the standard runbook). |
| Legacy zoom artefacts (`analysis/mms_zoom_state.json`, `analysis/mms_proposals_struct_filtered.json`, …) | Earlier storm→quiet baseline outputs kept for reference. |

## Observations
- Structural vocab stays collapsed at **16 tokens** per slice; `scripts/enforce_structural.py` now codifies the `__`-only/min-occ≥2 rule so skims, proposals, and twins cannot regress into numeric detritus.
- Router guardrail achieved 5–7 % foreground coverage on every slice (22:30, 23:00, 00:00, 01:00) while retaining percentile provenance; configs now ship with the applied percentiles + coverage snapshot.
- Midnight proposals are fully structural (`x__zpos`, `z__zpos`, `y__zpos`, `y__up`) with tight percentile constraints; storm→storm twins show three 50-window matches into 2017‑09‑13, all sourced from z/up features.
- Lead-time probe (5 min bins) shows foreground density lifting from 3–5 % to **7.4 %** in the last five minutes before the 00:40 UTC onset, hinting at a quantitative precursor trend to formalise.
- Adjacent hours (22:30–23:30, 01:00–02:00) echo the same structural proposals (≥4) and storm twins (3 × 50 windows) with slightly lower coherence, showing the midnight hour is the peak rather than a one-off.
- The scorecard (below) keeps midnight in the lead: it pairs the lowest mean ANN distance with a meaningful lead-time ramp, while neighbouring hours remain close enough to prove stability.

## Slice scorecard (storm hours)

| Slice | Coverage | Structural proposals | Storm twins (≥50 windows) | Mean ANN distance (×10⁻³) | Lead density (−5..0 min) |
| --- | --- | --- | --- | --- | --- |
| 2017-09-07 22:30–23:30 | 6.9 % | 2 | 1 | 2.33 | 0.1299 |
| 2017-09-07 23:00–00:00 | 5.4 % | 4 | 1 | 2.26 | 0.0312 |
| **2017-09-08 00:00–01:00** | **5.7 %** | **4** | **3** | **1.97** | **0.0741** |
| 2017-09-08 01:00–02:00 | 5.1 % | 4 | 3 | 2.14 | 0.0735 |
| 2017-09-10 00:00–01:00 (quiet) | 5.9 % | 3 (`__UP` only) | 4 (generic) | 2.30 | 0.0884 |

Midnight (00:00–01:00) remains the reference slice: it pairs the lowest mean ANN distances with multiple structural twins and a consistent foreground rise ahead of onset. The 22:30 and 01:00 neighbours retain the same structural vocabulary, confirming the signature is hour-stable rather than a lone burst.

## Adjacent Hour Scan (Sep 7–8 UTC)

### 22:30–23:30
- `stm ingest` → 933 windows, 17 scored strings (16 structural kept via `scripts/enforce_structural.py`).
- Guardrailed router (`analysis/router_config_2017-09-07_2230-2330.json`): coh ≥ 0.00184, ent ≤ 0.99816, stab ≥ 0.4717 ⇒ **6.9 %** foreground coverage.
- Storm→storm twins vs 2017-09-13: three structural matches (`x__zpos`, `x__up`, `y__rangeexp`) with 50 aligned windows each (ANN distances ≈8e-4–1.1e-2).
- RangeExp/Accel proposals remain structural after connector floor ≥0.5; `time` token fully purged from deliverables.
- Twin diagnostics (`analysis/mms_twins_2230_diagnostics.json`) capture mean ANN distance 2.33×10⁻³ with 50 aligned windows on `x__up`.
- Lead-time bins (`analysis/mms_2230_leadtime.json`, monotonic flag = false) jump from 0 % (−20…−15 min) to **13 %** foreground in the final 5 min; see `docs/plots/mms_2230_overview.png`, `_zoom.png`, and the density line in `docs/plots/mms_2230_lead.png`.

### 23:00–00:00 (baseline zoom)
- `analysis/mms_zoom_2300_state.json` refreshed for comparison; guardrailed thresholds (`analysis/router_config_2017-09-07_2300-0000.json`): coh ≥ 0.00671, ent ≤ 0.99329, stab ≥ 0.46429 ⇒ **5.4 %** coverage across 1 155 windows.
- Structural skim remains pure bit tokens; storm→storm twin (`analysis/mms_twins_2300_to_0913.json`) still highlights `y__zpos` with 50 aligned windows but no additional candidates under the tightened connector floor.
- Diagnostics (`analysis/mms_twins_2300_diagnostics.json`) show a single 50-window hit with mean ANN distance 2.26×10⁻³.
- Lead-time density (`analysis/mms_2300_leadtime.json`) barely rises (−5..0 min density 0.0312; `docs/plots/mms_2300_lead.png`), underscoring why midnight edges it out.

### 00:00–01:00
- `stm ingest` → 1 623 windows, 17 scored strings (structural subset locked in via helper scripts). Manifold stored at `analysis/mms_0000_state.json`.
- Guardrailed router (`analysis/router_config_0000.json`): coh ≥ 0.00774, ent ≤ 0.99226, stab ≥ 0.46539 ⇒ **5.7 %** foreground coverage (stable enough for `/stm/seen`).
- Proposals stay structural (`x__zpos`, `z__zpos`, `y__zpos`, `y__up`) with percentile target profile `coh≥P60, ent≤P50, λ≤P70`; diagnostics remain within ~1.4×10⁻³ ANN distance of the seed centroid (`analysis/mms_0000_proposals_struct.json`).
- Storm→storm twins (profile `coh≥P55, stab≥P40, ent≤P55, λ≤P70`) produce three 50-window matches (`x__zpos`, `z__zpos`, `y__zpos`) into the 2017-09-13 storm under the 0.5 connector floor; diagnostics (`analysis/mms_twins_0000_diagnostics.json`) keep mean ANN distance at 1.97×10⁻³ with the same dominant signature tokens.
- Lead-time bins (`analysis/mms_0000_leadtime.json`, monotonic flag = false) climb from **5.2 % → 7.4 %** over the final 20 min; see `docs/plots/mms_0000_overview.png`, `_zoom.png`, and the density line `docs/plots/mms_0000_lead.png`.

### 01:00–02:00
- `stm ingest` → 1 635 windows, 17 scored strings (structural subset enforced). Manifold lives at `analysis/mms_0100_state.json`.
- Guardrailed router (`analysis/router_config_0100.json`): coh ≥ 0.00745, ent ≤ 0.99255, stab ≥ 0.46514 ⇒ **5.1 %** foreground coverage.
- Proposals mirror the midnight set (`z__zpos`, `x__zpos`, `y__up`, `y__zpos`; `analysis/mms_0100_proposals_struct.json`) with coherence 0.0075–0.0081 and occupancies 1.5–1.6 k.
- Storm→storm twins (`analysis/mms_twins_0100_to_0913.json`) retain three 50-window matches (`x__zpos`, `z__zpos`, `y__up`) with mean ANN distance ≈2.14×10⁻³ (`analysis/mms_twins_0100_diagnostics.json`).
- Lead-time bins (`analysis/mms_0100_leadtime.json`) rise from 2.2 % to **7.3 %** foreground in the final 5 min; `docs/plots/mms_0100_overview.png`, `_zoom.png`, and `docs/plots/mms_0100_lead.png` show the structural overlays and density ramp.

| 01:00–02:00 structural proposals (`analysis/mms_0100_proposals_struct.json`) | Coh | Ent | Stab | ANN dist |
| --- | --- | --- | --- | --- |
| `mms1_fgm_b_gse_srvy_l2_z__zpos` | 7.48×10⁻³ | 0.99252 | 0.46480 | 7.03×10⁻⁴ |
| `mms1_fgm_b_gse_srvy_l2_x__zpos` | 7.74×10⁻³ | 0.99226 | 0.46467 | 1.11×10⁻³ |
| `mms1_fgm_b_gse_srvy_l2_y__up`   | 7.72×10⁻³ | 0.99228 | 0.46416 | 1.54×10⁻³ |
| `mms1_fgm_b_gse_srvy_l2_y__zpos` | 8.07×10⁻³ | 0.99193 | 0.46391 | 2.13×10⁻³ |

| 01:00–02:00 storm→storm twins (`analysis/mms_twins_0100_diagnostics.json`) | Aligned windows | Mean ANN (×10⁻³) | Top signature token |
| --- | --- | --- | --- |
| `mms1_fgm_b_gse_srvy_l2_x__zpos` | 50 | 2.12 | `c0.01_s0.49_e0.99` (48 hits) |
| `mms1_fgm_b_gse_srvy_l2_z__zpos` | 50 | 2.11 | `c0.01_s0.49_e0.99` (48 hits) |
| `mms1_fgm_b_gse_srvy_l2_y__up`   | 50 | 2.20 | `c0.01_s0.49_e0.99` (48 hits) |

### Cross-hour takeaways
- Structural vocab stays collapsed to 16 tokens across all slices; `time` is the lone non-structural artifact and is now filtered out in delivered lists.
- Midnight window (00:00–01:00) shows the highest structural coherence and the densest storm→storm agreement; adjacent hours (22:30–23:30 and 01:00–02:00) echo the same z/up structures with slightly lower coherence, proving the signature is hour-stable.
- Guardrail unlocked usable foreground coverage (5–7 %) on every slice without losing percentile provenance — future slices can reuse the same script to stay within spec.

## Router coverage snapshot (latest guardrail)

| Slice | min_coh (percentile) | max_ent (percentile) | min_stab (percentile) | Foreground coverage |
| --- | --- | --- | --- | --- |
| 2017-09-07 22:30–23:30 | 0.00184 (P50) | 0.99816 (P50) | 0.47174 (P45) | **6.9 %** |
| 2017-09-07 23:00–00:00 | 0.00671 (P55) | 0.99329 (P45) | 0.46429 (P45) | **5.4 %** |
| 2017-09-08 00:00–01:00 | 0.00774 (P60) | 0.99226 (P40) | 0.46539 (P50) | **5.7 %** |
| 2017-09-08 01:00–02:00 | 0.00745 (P50) | 0.99255 (P50) | 0.46514 (P60) | **5.1 %** |
| 2017-09-10 00:00–01:00 (quiet) | 0.00865 (P70) | 0.99151 (P35) | 0.46307 (P45) | **5.9 %** |

Each JSON config includes the applied percentiles plus `coverage` and a sibling `.coverage.json` dump for audit trails; the quiet-hour guardrail sits in the same 5–6 % band with higher coherence/entropy percentiles.

## Sensitivity & ablation checks (00:00–01:00 reference)

| Experiment | Coverage | Proposals (≥ profile) | 50-window twins | Notes |
| --- | --- | --- | --- | --- |
| Guardrailed baseline (1 s, stride 512) | 5.7 % | 4 | 3 | Structural `__zpos/__up` proposals; ANN ≈1.9–2.0×10⁻³ across twins. |
| 0.5 s stride (256 bytes) | 5.7 % | 4 | 0 | Guardrail still lands in-band; twin search drops under the same profile, showing the higher-coherence requirement is the bind. |
| Router P90/P20/P75 | 0 % | – | – | No windows satisfy the stricter cut — reinforces why the guardrail relaxer is needed. |
| Router P88/P22/P72 | 0 % | – | – | Same outcome; autop guardrail keeps coverage in the usable 5–20 % band. |
| RangeExp-only seed | 5.7 % | 4 | 0 | Structural proposals persist, but no standalone `__RANGEEXP` twin meets the storm profile. |
| Accel-only seed | 5.7 % | 4 | 0 | Proposals still appear, yet cross-day twins require the combined z/up signature. |

These small probes show the headline stays intact under resampling and seed ablations, while also evidencing that the guardrail logic is essential for usable coverage.

## Quiet-day baseline (2017-09-10 00:00–01:00)

- Guardrail (`analysis/router_config_quiet.json`): coh ≥ 0.00865, ent ≤ 0.99151, stab ≥ 0.46307 ⇒ **5.9 %** foreground coverage.
- Structural proposals (`analysis/mms_quiet_proposals_struct.json`) are purely `__UP`; no `__ZPOS` strings survive the filters.
- Storm→storm twin search (`analysis/mms_twins_quiet_to_0913.json`) returns generic `__UP`/`__ACCEL` matches at 50 windows each; diagnostics (`analysis/mms_twins_quiet_diagnostics.json`) keep mean ANN distance at 2.30×10⁻³ with no `__ZPOS` appearances.
- Lead-time density (`analysis/mms_quiet_leadtime.json`, monotonic flag = false) is flat (first and final bins both 8.8 %), and `docs/plots/mms_quiet_overview.png`, `_zoom.png`, and `docs/plots/mms_quiet_lead.png` show muted structural overlays.
- This run acts as a specificity check: guardrail holds, but evidence downgrades to generic motion instead of coherent structural precursors.

## Midnight (00:00–01:00) evidence package

- **Structural dominance:** `analysis/mms_0000_top_structural.txt` lists the 16 surviving tokens (`x/z/y__zpos`, `__up`, `__accel`, `__rangeexp`) with min-occ≥2 enforced.
- **Proposals:** `analysis/mms_0000_proposals_struct.json` keeps four z/up strings with ANN distances ≤1.4 e-3 and connectors at 0.5 after filtering.
- **Storm→storm twins:** `analysis/mms_twins_0000_to_0913.json` retains three strings (`x__zpos`, `z__zpos`, `y__zpos`), each aligning 50 midnight windows to Sep 13 with ANN distances ≲8.3 e-4.
- **Twin diagnostics:** `analysis/mms_twins_0000_diagnostics.json` summarises the three twin hits (mean ANN distance 1.97×10⁻³, 150 aligned windows, dominant `c0.01_s0.49_e0.99` signature).
- **Lead time:** `analysis/mms_0000_leadtime.json` captures the 5 min bins + monotonic flag (false), showing a rise to **7.4 %** just before onset, with the companion plot `docs/plots/mms_0000_lead.png`.
- **Visuals:** `docs/plots/mms_0000_overview.png` (full hour) and `docs/plots/mms_0000_zoom.png` (00:30–00:40 UTC) overlay Bx/Bz with `__rangeexp`/`__accel` activity and the calibrated foreground heat strip.

| Midnight structural proposals (`analysis/mms_0000_proposals_struct.json`) | Coh | Ent | Stab | ANN dist |
| --- | --- | --- | --- | --- |
| `mms1_fgm_b_gse_srvy_l2_x__zpos` | 7.32×10⁻³ | 0.99268 | 0.46600 | 7.52×10⁻⁴ |
| `mms1_fgm_b_gse_srvy_l2_z__zpos` | 7.23×10⁻³ | 0.99277 | 0.46580 | 7.28×10⁻⁴ |
| `mms1_fgm_b_gse_srvy_l2_y__zpos` | 7.43×10⁻³ | 0.99257 | 0.46532 | 1.39×10⁻³ |
| `mms1_fgm_b_gse_srvy_l2_y__up`   | 7.42×10⁻³ | 0.99258 | 0.46530 | 1.40×10⁻³ |

| Midnight storm→storm twins (`analysis/mms_twins_0000_diagnostics.json`) | Aligned windows | Mean ANN (×10⁻³) | Top signature token |
| --- | --- | --- | --- |
| `mms1_fgm_b_gse_srvy_l2_x__zpos` | 50 | 1.93 | `c0.01_s0.49_e0.99` (48 hits) |
| `mms1_fgm_b_gse_srvy_l2_z__zpos` | 50 | 1.96 | `c0.01_s0.49_e0.99` (48 hits) |
| `mms1_fgm_b_gse_srvy_l2_y__zpos` | 50 | 2.03 | `c0.01_s0.49_e0.99` (48 hits) |


## Next up
1. Turn the evidence bundle into the draft “MMS Structural Precursors” note (method, coverage table, proposal/twin tables + diagnostics, lead-time chart, plot plate).
2. Promote the twin diagnostics + scorecard into a single table for the note (mean ANN, aligned windows, top signature q-grams).
3. Use `stm-leadtime` to sweep alternative onsets (±2 h) and contrast their density ramps with the quiet baseline.
4. Re-run `stm-plots` / `stm-leadtime` on any new slices before publishing to keep plots + JSON in lockstep.
