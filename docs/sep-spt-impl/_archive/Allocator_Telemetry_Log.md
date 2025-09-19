# Allocator Telemetry Log

Purpose: Track allocator-lite + rolling evaluator behaviour using Prometheus exports and Valkey snapshots. Update entries whenever new artifacts land in `docs/reports/`.

_Last updated: 2025-09-16 09:50 UTC (based on `docs/reports/live_runs/2025-09-15_live_digest.md` and `docs/reports/snapshots/`)._

## Latest Snapshot
- **Data sources**
  - Prometheus scrape (`docs/reports/live_runs/2025-09-15_live_digest.md` → `sep_rolling_gate`, `sep_allocator_selected`, `sep_nav_*`).
  - Valkey dumps (`docs/reports/snapshots/snap_alloc.json`, `manifold_metrics_all.csv`, `manifold_metrics_samples.csv`).
- **Eligible count** — 6 instruments (`sep_rolling_gate{eligible="1"}`) with buffers: EUR_USD 0.10, USD_JPY 0.10, EUR_JPY 0.10, USD_CAD 0.10, NZD_USD 0.12, USD_CHF 0.15.
- **Top-K selection** — allocator-lite weights from `snap_alloc.json`: {USD_JPY 0.118, EUR_USD 0.103, EUR_JPY 0.101}; total allocation 0.322 (≈ target 30% allowing ramp buffer).
- **NAV vs Calmar** — 7d NAV change +2.1% while strict Calmar of selected instruments averages 0.42 (reference `live_digest.md`). Divergence within expected bounds.
- **Freshness** — `opt:rolling:gates_blob.ts` 2025-09-15T23:49:20Z, lag 42s vs evaluator cadence (healthy).

## Metrics Monitored
| Signal | Source | Threshold / Expectation |
| --- | --- | --- |
| `sep_rolling_gate{eligible="1"}` | Prometheus `/metrics` | ≥3 during active sessions |
| `sep_allocator_selected{}` count | Prometheus | Equals `ALLOC_TOP_K` |
| `sep_allocator_cooldown{}` | Prometheus | Investigate if ≥5 instruments for ≥30m |
| `sep_nav_change_7d` vs Calmar | Prometheus & `bt:rolling:summary` | Divergence >0.0 while Calmar >0.1 triggers exit audit |
| `risk:allocation_weights` | Valkey snapshot | Updated every `ALLOC_INTERVAL_SEC`; weights sum ≈ total allocation |
| `opt:rolling:gates_blob.ts` | Valkey snapshot | Age < `GATES_BLOB_MAX_AGE_SEC` |

## Refresh Procedure
1. Dump Prometheus metrics: `curl -s :8000/metrics > docs/reports/live_runs/$(date +%Y-%m-%d)_live_digest.prom` and summarise into Markdown similar to `2025-09-15_live_digest.md`.
2. Capture allocation snapshot: `docker exec -i sep-valkey redis-cli GET risk:allocation_weights | jq > docs/reports/snapshots/snap_alloc.json`.
3. Export manifold metrics: `docker exec -i sep-valkey redis-cli --raw HGETALL manifold:metrics:latest` (if enabled) or run `docs/scripts/export_metrics.py` (TODO) to update CSVs.
4. Update the **Latest Snapshot** section above with timestamp, eligible counts, weights, and NAV vs Calmar commentary.
5. Append notable incidents or parameter changes under Observations.

## Observations & Actions
- 2025-09-15: Eligible count dipped to 2 during late New York session because NZD_USD calmar < floor; allocator-lite held previous weights — no manual intervention required.
- 2025-09-16 00:10Z: `opt:rolling:gates_blob` lagged 3 intervals due to delayed candle; alert fired and auto-cleared after evaluator catch-up.

## Upcoming Checks
- Validate λ hazard penalty once `ALLOC_SCORE_WEIGHT_HAZARD` rolls out (see `program/plan.md`).
- Add automated exporter to drop Prometheus summaries into `docs/reports/live_runs/latest.prom` nightly for consistent diffs.
