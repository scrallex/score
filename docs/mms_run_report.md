# MMS Structural Manifold – Initial Findings

## Setup
- **Data**: MMS1 FGM (magnetic field) + FPI (plasma moments) for 2017-09-07–09, 10, 13.
- **Preprocessing**:
  - Native sampling left at 1 s cadence.
  - `scripts/encode_struct_bits.py` converts each numeric channel into structural bit streams (`__UP`, `__ACCEL`, `__RANGEEXP`, `__ZPOS`).
  - Zoom corpus (`nasa/mms/csv_zoom/2017-09-07_22-02`) keeps only the 22:00–02:00 UTC storm window.
- **Kernel**: STM native (QFH/QBSA) with `window_bytes=1024`, `stride=512`, `--drop-numeric` to focus on bit tokens.
- **Indices**: `stm index build` → `analysis/mms_zoom_ann.hnsw`, `analysis/mms_zoom_postings.json`.
- **Router calibration**: `scripts/calibrate_router.py` now enforces a 5–20 % foreground coverage band by relaxing percentile cuts in a fixed order and records the effective thresholds + coverage per slice.

## Current Deliverables
| Artifact | Purpose |
| --- | --- |
| `analysis/mms_0000_state.json` | Midnight (00:00–01:00) structural manifold (`--drop-numeric`, signals, profiles stored). |
| `analysis/router_config_0000.json` + `.coverage.json` | Guardrailed thresholds for midnight slice (coverage ≈5.7 %, percentile record included). |
| `analysis/mms_2017-09-07_2230-2330_state.json` / `analysis/router_config_2017-09-07_2230-2330.json` | 22:30–23:30 manifold + retuned router (coverage ≈6.9 %). |
| `analysis/mms_zoom_2300_state.json` / `analysis/router_config_2017-09-07_2300-0000.json` | 23:00–00:00 baseline manifold + retuned router (coverage ≈5.4 %). |
| `analysis/mms_*_top_structural.txt` | Structural-only top strings (`scripts/enforce_structural.py strings … --min-occ 2`). |
| `analysis/mms_0000_proposals_struct.json` | Four structural proposals seeded by `x__rangeexp`/`z__accel` under tight percentiles. |
| `analysis/mms_twins_0000_to_0913.json` | Storm→storm twins (midnight → 2017-09-13) with 3 structural strings × 50 aligned windows. |
| `analysis/mms_0000_leadtime.json` | 5 min lead-time bins showing foreground density climb toward onset. |
| `docs/plots/mms_0000_overview.png`, `docs/plots/mms_0000_zoom.png` | Bx/Bz traces with `__rangeexp`/`__accel` overlays + foreground heat strip (full hour + focused 10 min). |
| `scripts/enforce_structural.py` | CLI helper to enforce `__` tokens + connector floor for strings/proposals/twins. |
| `scripts/plot_structural_window.py` | Plot generator for structural overlays (used for the PNGs above). |
| `scripts/lead_time_density.py` | Foreground density vs lead time (bin-based precursor probe). |
| Legacy zoom artefacts (`analysis/mms_zoom_state.json`, `analysis/mms_proposals_struct_filtered.json`, …) | Earlier storm→quiet baseline outputs kept for reference. |

## Observations
- Structural vocab stays collapsed at **16 tokens** per slice; `scripts/enforce_structural.py` now codifies the `__`-only/min-occ≥2 rule so skims, proposals, and twins cannot regress into numeric detritus.
- Router guardrail achieved 5–7 % foreground coverage on every slice (22:30, 23:00, 00:00) while retaining percentile provenance; configs now ship with the applied percentiles + coverage snapshot.
- Midnight proposals are fully structural (`x__zpos`, `z__zpos`, `y__zpos`, `y__up`) with tight percentile constraints; storm→storm twins show three 50-window matches into 2017‑09‑13, all sourced from z/up features.
- Lead-time probe (5 min bins) shows foreground density lifting from 3–5 % to **7.4 %** in the last five minutes before the 00:40 UTC onset, hinting at a quantitative precursor trend to formalise.

## Adjacent Hour Scan (Sep 7–8 UTC)

### 22:30–23:30
- `stm ingest` → 933 windows, 17 scored strings (16 structural kept via `scripts/enforce_structural.py`).
- Guardrailed router (`analysis/router_config_2017-09-07_2230-2330.json`): coh ≥ 0.00184, ent ≤ 0.99816, stab ≥ 0.4717 ⇒ **6.9 %** foreground coverage.
- Storm→storm twins vs 2017-09-13: three structural matches (`x__zpos`, `x__up`, `y__rangeexp`) with 50 aligned windows each (ANN distances ≈8e-4–1.1e-2).
- RangeExp/Accel proposals remain structural after connector floor ≥0.5; `time` token fully purged from deliverables.

### 23:00–00:00 (baseline zoom)
- `analysis/mms_zoom_2300_state.json` refreshed for comparison; guardrailed thresholds (`analysis/router_config_2017-09-07_2300-0000.json`): coh ≥ 0.00671, ent ≤ 0.99329, stab ≥ 0.46429 ⇒ **5.4 %** coverage across 1 155 windows.
- Structural skim remains pure bit tokens; storm→storm twin (`analysis/mms_twins_2300_to_0913.json`) still highlights `y__zpos` with 50 aligned windows but no additional candidates under the tightened connector floor.

### 00:00–01:00
- `stm ingest` → 1 623 windows, 17 scored strings (structural subset locked in via helper scripts). Manifold stored at `analysis/mms_0000_state.json` for the re-ingest pass.
- Guardrailed router (`analysis/router_config_0000.json`): coh ≥ 0.00774, ent ≤ 0.99226, stab ≥ 0.46539 ⇒ **5.7 %** foreground coverage (stable enough for `/stm/seen`).
- Proposals stay structural (`x__zpos`, `z__zpos`, `y__zpos`, `y__up`) with percentile target profile `coh≥P60, ent≤P50, λ≤P70`; diagnostics remain within ~1.4e-3 ANN distance of the seed centroid.
- Storm→storm twins (profile `coh≥P55, stab≥P40, ent≤P55, λ≤P70`) produce three 50-window matches (`x__zpos`, `z__zpos`, `y__zpos`) into the 2017-09-13 storm under the 0.5 connector floor.
- Visual QA: `docs/plots/mms_0000_overview.png` + `_zoom.png` overlay Bx/Bz with `__rangeexp` / `__accel` flags alongside the calibrated foreground heat strip.

### Cross-hour takeaways
- Structural vocab stays collapsed to 16 tokens across all slices; `time` is the lone non-structural artifact and is now filtered out in delivered lists.
- Midnight window (00:00–01:00) shows the highest structural coherence and the densest storm→storm agreement; 22:30–23:30 still carries distinct RangeExp spikes worth folding into ensembles.
- Guardrail unlocked usable foreground coverage (5–7 %) on every slice without losing percentile provenance — future slices can reuse the same script to stay within spec.

## Router coverage snapshot (latest guardrail)

| Slice | min_coh (percentile) | max_ent (percentile) | min_stab (percentile) | Foreground coverage |
| --- | --- | --- | --- | --- |
| 2017-09-07 22:30–23:30 | 0.00184 (P50) | 0.99816 (P50) | 0.47174 (P45) | **6.9 %** |
| 2017-09-07 23:00–00:00 | 0.00671 (P55) | 0.99329 (P45) | 0.46429 (P45) | **5.4 %** |
| 2017-09-08 00:00–01:00 | 0.00774 (P60) | 0.99226 (P40) | 0.46539 (P50) | **5.7 %** |

Each JSON config includes the applied percentiles plus `coverage` and a sibling `.coverage.json` dump for audit trails.

## Midnight (00:00–01:00) evidence package

- **Structural dominance:** `analysis/mms_0000_top_structural.txt` lists the 16 surviving tokens (`x/z/y__zpos`, `__up`, `__accel`, `__rangeexp`) with min-occ≥2 enforced.
- **Proposals:** `analysis/mms_0000_proposals_struct.json` keeps four z/up strings with ANN distances ≤1.4 e-3 and connectors at 0.5 after filtering.
- **Storm→storm twins:** `analysis/mms_twins_0000_to_0913.json` retains three strings (`x__zpos`, `z__zpos`, `y__zpos`), each aligning 50 midnight windows to Sep 13 with ANN distances ≲8.3 e-4.
- **Lead time:** `analysis/mms_0000_leadtime.json` shows foreground density lifting from 3–5 % (−20…−5 min) to **7.4 %** in the final five minutes before the 00:40 UTC onset.
- **Visuals:** `docs/plots/mms_0000_overview.png` (full hour) and `docs/plots/mms_0000_zoom.png` (00:30–00:40 UTC) overlay Bx/Bz with `__rangeexp`/`__accel` activity and the calibrated foreground heat strip.

## Next up
1. Extend the guardrail workflow to additional slices (e.g. 22:30–23:30, 00:00–01:00 ±1 h) and log the thresholds + coverage for the note.
2. Quantify storm→storm alignment quality (mean distance, q-gram overlaps) for the midnight twins to turn the 50-window evidence into a headline table.
3. Expand the lead-time probe to adjacent hours and alternative onsets to see if the density ramp generalises beyond the 00:40 UTC event.
4. Draft the “MMS Structural Precursors” note: method recap, coverage table, proposal/twin highlights, lead-time plot, and the two structural overlays.
