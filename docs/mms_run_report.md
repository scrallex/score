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
| `analysis/router_config.json` | Router thresholds tuned to MMS percentile bands. |
| `analysis/mms_zoom_top_structural.txt` | High-signal structural strings filtered by `coh>=P50,ent<=P80`. |
| `analysis/mms_proposals_struct_filtered.json` | Proposal example: `mms1_fgm_b_gse_srvy_l2_mag__zpos` with diagnostics. |
| `analysis/mms_twins_zoom_0708_to_10.json` | Storm→quiet twin report (10 matches, ANN+q-gram evidence). |
| `scripts/encode_struct_bits.py`, `scripts/calibrate_router.py` | Reusable preprocessing + calibration tooling. |

## Observations
- Structural vocab collapsed from ~300 k raw tokens to **60** structural tokens once bit features & `--drop-numeric` are used. Bit-prefixed strings (`mms1_fgm_*__RANGEEXP`, `__ACCEL`, …) now dominate every ranking.
- Router foreground is now relative: `/stm/seen` for `mms1_fgm_b_gse_srvy_l2_x__rangeexp` returns high-coherence/low-entropy windows despite absolute coherence ≈0.01.
- First structural proposal (`mms1_fgm_b_gse_srvy_l2_mag__zpos`) survived percentile filters `lambda<=P70, coh>=P60, ent<=P50`.
- Storm→quiet twin search finds 10 matches; remaining numeric residues highlight the need to enforce bit-prefix filtering on future comparisons.

## Next up
1. **Micro-slices** (≤1 h) centred on onset; re-ingest & recalibrate per slice.
2. **Structural-only filters** for CLI/proposals/twins (require `__` prefix or `min_occ` guards) to suppress lingering scientific-notation tokens.
3. **Storm→storm** twin comparison (e.g. Sep 7/8 vs Sep 13) to show structural repetition across events.
4. **QA plots**: overlay RANGEEXP/ACCEL flags on Bx/Bz traces; visual foreground heatmap to confirm calibration behaviour.
