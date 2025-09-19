# 02. Operations Runbook

This runbook collects the operational procedures that keep the live quantum-metrics stack healthy. Each section is self-contained; cross-reference the numbered core docs for architecture (01), developer guidance (04), and pipeline details (09, 11).

## Pre-Open Spin-Up (≈5–10 minutes)
1. **Kill switch & live flags** – confirm `/api/kill-switch → {kill_switch:false}` and `LIVE_TRADING_ENABLED=1`. Use `KILL_SWITCH_FORCE_ENV=1` only if the environment must override Valkey on boot.
2. **Freshness** – ensure `ws:last:manifold:{instrument}` timestamps are ≤30 seconds old and `sep:signal_index:{instrument}` shows inserts within the last 5 minutes. The deploy warmup orchestrator primes both during boot.
3. **Gates blob** – inspect `opt:rolling:gates_blob.ts` and `/metrics sep_rolling_gate{eligible="1"}`; expect at least three eligibles for the active sessions. Confirm repetition counts and hazards look sane (`gates_blob.repetition` and `gates_blob.hazard`).
4. **Sessions & margin** – confirm the intended sessions are open (Tokyo/London/New York) and margin utilisation `< 0.30` via `/api/oanda/account`.
5. **Allocator & portfolio** – allocator-lite publishes ≥`ALLOC_TOP_K` weights each minute (`/metrics sep_allocator_selected`). PortfolioManager ramps toward caps without exceeding `min(budget/K, PM_MAX_PER_POS_PCT × budget)`.
6. **If still flat after 5 minutes** – inspect OANDA connectivity, spread/margin guards, gates freshness, and allocator-lite logs before adjusting parameters.

## Daily Ops Snapshot (≈30 seconds)
1. `curl :8100/status | jq .interval` → expect `60` (allocator-lite cadence).
2. `/metrics` – review `sep_rolling_gate{}` rows to confirm eligibles and buffers per instrument.
3. Count `sep_allocator_selected{}` – ensures the published weights match `ALLOC_TOP_K`.
4. `opt:rolling:gates_blob.ts` age within `GATES_BLOB_MAX_AGE_SEC`.
5. `sep_span_integrity_ok` metric equals `1`.
6. `/api/risk/allocation-status` – total allocation ≤0.30 and weights present only for the published Top-K.

## Post-Deploy Warmup
1. Run `deploy.sh` – builds images, starts services, and executes warmup orchestrator (`backfill`, `prime`, `hydrate`).
2. Verify candles via `/api/candles?instrument=EUR_USD&granularity=M1&limit=120` and manifolds via `redis-cli GET ws:last:manifold:EUR_USD`.
3. Confirm allocator-lite (port 8100) and rolling evaluator logs are healthy before disengaging the kill switch. Env knobs: `ECHO_MIN_REPETITIONS`, `ECHO_HAZARD_MAX`.

## Troubleshooting: No Positions While Gates Are Green
1. Kill switch off? (`/api/kill-switch → false`).
2. Trading mode live? (`/api/trading/mode → mode:"live", oanda_connected:true`).
3. Budget populated? (`redis-cli HGETALL risk:alloc_budget`).
4. Allocation weights exist? (`redis-cli GET risk:allocation_weights`).
5. Freshness within limits? (`ws:last:manifold:*`, `sep:signal_index:*`).
6. Gates blob stale? Allocator-lite holds previous weights until refreshed; **do not** force entries.
7. Spread / margin gating blocking adds? Inspect `/api/oanda/account` and allocator logs for `MAX_SPREAD_BPS` or margin warnings.
8. If all above pass, trigger manual reconcile: `POST /api/trade/reconcile`. PortfolioManager also reconciles automatically every 30 seconds.

## Recovery: Margin or Exposure Beyond 30%
1. Engage kill switch (blocks new adds, allows reductions).
2. Run `bin/drain_overcap_only.sh --dry-run` to inspect proposed closures.
3. Close positions with a ≤3% buffer until utilisation `< 0.30`.
4. Review `RISK_ALLOC_TARGET_PCT`, Top-K selection, and live spreads before re-enabling entries.

## Weekend / Market Close
1. Expect WS ages to stretch; allocator-lite keeps the previous weights while sessions are closed.
2. Leave the kill switch ON until the first session opens and freshness checks pass.
3. Resume with the Pre-Open Spin-Up checklist.

Keep this file under version control alongside the code; any operational knob change should update the relevant checklist.
