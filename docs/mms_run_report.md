# MMS Structural Manifold – Initial Findings

## Setup
- **Data**: MMS1 FGM (magnetic field) + FPI (plasma moments) for 2017-09-07–09, 10, 13.
- **Preprocessing**:
  - Native sampling left at 1 s cadence.
  - `scripts/encode_struct_bits.py` converts each numeric channel into structural bit streams (`__UP`, `__ACCEL`, `__RANGEEXP`, `__ZPOS`).
  - Zoom corpus (`nasa/mms/csv_zoom/2017-09-07_22-02`) keeps only the 22:00–02:00 UTC storm window.
- **Kernel**: STM native (QFH/QBSA) with `window_bytes=1024`, `stride=512`, `--drop-numeric` to focus on bit tokens.
- **Indices**: `stm index build` → `analysis/mms_zoom_ann.hnsw`, `analysis/mms_zoom_postings.json`.
- **Router calibration**: `scripts/calibrate_router.py` on `analysis/mms_zoom_state.json` → coherence ≥ P75 (≈0.0102), entropy ≤ P35 (≈0.991), stability ≥ P60 (≈0.465).

## Current Deliverables
| Artifact | Purpose |
| --- | --- |
| `analysis/mms_zoom_state.json` | Zoom manifold (signals, profiles, percentile-calibrated metrics). |
| `analysis/router_config.json` | Router thresholds tuned to MMS percentile bands (latest run). |
| `analysis/mms_zoom_top_structural.txt` | High-signal structural strings filtered by `coh>=P50,ent<=P80`. |
| `analysis/mms_proposals_struct_filtered.json` | Proposal example: `mms1_fgm_b_gse_srvy_l2_mag__zpos` with diagnostics. |
| `analysis/mms_twins_zoom_0708_to_10.json` | Storm→quiet twin report (10 matches, ANN+q-gram evidence). |
| `analysis/mms_2017-09-07_2230-2330_state.json` | 22:30–23:30 zoom manifold (`--drop-numeric`, signals stored for router tuning). |
| `analysis/mms_2017-09-08_0000-0100_state.json` | 00:00–01:00 zoom manifold (higher-coherence structural slice). |
| `analysis/router_config_2017-09-07_2230-2330.json` | Percentile thresholds for the 22:30–23:30 foreground pass. |
| `analysis/router_config_2017-09-07_2300-0000.json` | Router thresholds for the 23:00–00:00 baseline slice. |
| `analysis/router_config_2017-09-08_0000-0100.json` | Percentile thresholds for the 00:00–01:00 foreground pass. |
| `analysis/mms_2017-09-07_2230-2330_top_structural.txt` | Structural-only top strings (pattern-sorted, `grep "__"`). |
| `analysis/mms_2017-09-08_0000-0100_top_structural.txt` | Structural-only top strings for the midnight slice. |
| `analysis/mms_proposals_struct_2017-09-07_2230-2330.json` | RangeExp/Accel-seeded proposals (structural filtered). |
| `analysis/mms_proposals_struct_2017-09-08_0000-0100.json` | Midnight proposals after structural gating. |
| `analysis/mms_twins_2017-09-07_2230-2330_to_0913.json` | Storm→storm twins (Sep 7 22:30–23:30 → Sep 13) with 3 structural strings. |
| `analysis/mms_twins_2017-09-08_0000-0100_to_0913.json` | Storm→storm twins (Sep 8 00:00–01:00 → Sep 13) with 4 structural strings. |
| `scripts/encode_struct_bits.py`, `scripts/calibrate_router.py` | Reusable preprocessing + calibration tooling. |

## Observations
- Structural vocab collapsed from ~300 k raw tokens to **60** structural tokens once bit features & `--drop-numeric` are used. Bit-prefixed strings (`mms1_fgm_*__RANGEEXP`, `__ACCEL`, …) now dominate every ranking.
- Router foreground is now relative: `/stm/seen` for `mms1_fgm_b_gse_srvy_l2_x__rangeexp` returns high-coherence/low-entropy windows despite absolute coherence ≈0.01.
- First structural proposal (`mms1_fgm_b_gse_srvy_l2_mag__zpos`) survived percentile filters `lambda<=P70, coh>=P60, ent<=P50`.
- Storm→quiet twin search finds 10 matches; remaining numeric residues highlight the need to enforce bit-prefix filtering on future comparisons.

## Adjacent Hour Scan (Sep 7–8 UTC)

### 22:30–23:30
- `stm ingest` → 933 windows, 17 scored strings (16 structural after `__` filter); manifold stored at `analysis/mms_2017-09-07_2230-2330_state.json`.
- Router quantiles (`analysis/router_config_2017-09-07_2230-2330.json`): coh ≥ 0.00635, ent ≤ 0.99727, stab ≥ 0.47430. Foreground coverage hits 0 / 933 windows → thresholds need loosening before live routing.
- Storm→storm twins vs 2017-09-13: three structural matches (`mms1_fgm_b_gse_srvy_l2_x__zpos`, `__x__up`, `__y__rangeexp`), each retrieving 50 aligned windows with ANN distances ≈9e-4–1.1e-2.
- Seeded proposals (`analysis/mms_proposals_struct_2017-09-07_2230-2330.json`) surface `x__zpos`, `y__rangeexp`, `x__up`, all connector ≥0.50; top strings file keeps `time` suppressed via `grep "__"`.

### 23:00–00:00 (baseline zoom)
- `analysis/mms_zoom_2300_state.json` refreshed for comparison; thresholds (`analysis/router_config_2017-09-07_2300-0000.json`): coh ≥ 0.00815, ent ≤ 0.99258, stab ≥ 0.46715. Coverage is 0 / 1155 windows, mirroring the earlier calibration tightness.
- Structural skim remains pure bit tokens; existing twin report (`analysis/mms_twins_2300_to_0913.json`) yields a single high-quality string (`mms1_fgm_b_gse_srvy_l2_y__zpos`, 50 matches).

### 00:00–01:00
- `stm ingest` → 1623 windows, 17 scored strings (16 structural). Structural coherence lifts to 0.00715 mean; results saved at `analysis/mms_2017-09-08_0000-0100_state.json`.
- Router quantiles (`analysis/router_config_2017-09-08_0000-0100.json`): coh ≥ 0.00847, ent ≤ 0.99204, stab ≥ 0.46637. Foreground coverage modest (6 / 1623 ≈ 0.37 %), still below the 5–20 % guardrail but non-zero.
- Storm→storm twins deliver four structural candidates (`x__zpos`, `x__up`, `z__zpos`, `y__zpos`) with consistent 50-window matches → strongest cross-day echo so far.
- Structural proposals (`analysis/mms_proposals_struct_2017-09-08_0000-0100.json`) rank the same z/up strings, corroborating the twin hits; top-structural skim expanded with RangeExp occurrences ≥700.

### Cross-hour takeaways
- Structural vocab stays collapsed to 16 tokens across all slices; `time` is the lone non-structural artifact and is now filtered out in delivered lists.
- Midnight window (00:00–01:00) shows the highest structural coherence and the densest storm→storm agreement; 22:30–23:30 carries distinct RangeExp spikes worth preserving for ensembles.
- Router calibration via pure percentile cuts is overly strict (0–0.37 % coverage). Next tuning pass should either relax the stability percentile or incorporate a minimum-coverage heuristic.

## Next up
1. **Visual QA plots** for the chosen hour (Bx/Bz with `__RANGEEXP`/`__ACCEL` overlays + foreground heat strip) → grounds the percentile choices.
2. **Storm→storm twins (best hour)**: rerun `stm similar` with tightened connector/percentile profile on `analysis/mms_2017-09-08_0000-0100_state.json` and compare against Sep 13.
3. **Lead-time probe**: slide 10 min bins backward from the (eyeballed) onset and quantify foreground density to test for precursor lift.
4. **Router tuning**: adjust calibration (e.g. swap stability percentile to P50 or add minimum-hit clamp) so each slice keeps 5–20 % windows in the foreground.
