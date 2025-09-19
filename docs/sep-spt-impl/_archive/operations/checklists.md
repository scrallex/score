# Operations Checklists

These checklists keep daily operations focused on the quantum-metrics stack. Run them verbatim; cross-reference `operations/reference.md` when you need command syntax or deeper context.

## Pre-Open Spin-Up (5–10 minutes)
1. **Kill switch & live flags** — confirm `LIVE_TRADING_ENABLED=1` and kill switch OFF (`/api/kill-switch → {kill_switch:false}`); set `KILL_SWITCH_FORCE_ENV=1` only if the env must override Valkey on boot.
2. **Freshness** — verify `ws:last:manifold:{pair}` timestamps ≤ 30s and `sep:signal_index:{pair}` has inserts within the last 5 minutes (warmup orchestrator primes these during deploy).
3. **Gates blob** — `opt:rolling:gates_blob.ts` is recent and `/metrics` reports ≥3 eligibles (`sep_rolling_gate{eligible="1"}`) for the current session.
4. **Sessions & margin** — confirm target sessions open (Tokyo/London/New York). Margin utilisation `< 0.30` using `/api/oanda/account`.
5. **Allocator & portfolio** — allocator-lite publishes ≥K weights each minute (`/metrics` `sep_allocator_selected` count = `ALLOC_TOP_K`). PortfolioManager ramps toward caps without exceeding `min(budget/K, PM_MAX_PER_POS_PCT × budget)`.
6. **If still flat after 5 minutes** — inspect OANDA connectivity, spread/margin guards, gates freshness, and allocator-lite logs before touching parameters.

## Daily Ops Snapshot (≈30 seconds)
1. `curl :8100/status | jq .interval_sec` → expect `60` (allocator-lite cadence).
2. `/metrics` extract `sep_rolling_gate` rows → confirm eligibles and buffers per instrument.
3. Count `sep_allocator_selected{}` → equals `ALLOC_TOP_K`.
4. Check `opt:rolling:gates_blob.ts` age (`redis-cli GET ... | jq .ts`) within `GATES_BLOB_MAX_AGE_SEC`.
5. Ensure `sep_span_integrity_ok` metric equals `1`.
6. `/api/risk/allocation-status` → total allocation ≤ 0.30 with weights on the published Top-K only.

## Post-Deploy Warmup
1. Run `deploy.sh` (ensures services + warmup orchestrator).
2. Verify warmup orchestrator primes candles (`/api/candles?limit=120`) and hydrates manifolds (WS mirrors populated).
3. Confirm allocator-lite + rolling evaluator are healthy before disengaging kill switch.

## Troubleshooting: No Positions While Gates Are Green
1. Kill switch off? (`/api/kill-switch → false`).
2. Trading mode live? (`/api/trading/mode → mode:"live", oanda_connected:true`).
3. Budget populated? (`redis-cli HGETALL risk:alloc_budget`).
4. Allocation weights exist? (`redis-cli GET risk:allocation_weights`).
5. Freshness within limits? (`ws:last:manifold:*`, `sep:signal_index:*`).
6. Gates blob stale? Hold previous weights until refreshed; do **not** force fills.
7. Spread / margin gating blocking entries? Inspect `/api/oanda/account` and `MAX_SPREAD_BPS` logs.
8. If all above pass, trigger manual reconcile: `POST /api/trade/reconcile` (PortfolioManager loop runs automatically every 30s).

## Recovery: Margin or Exposure Beyond 30%
1. Engage kill switch (blocks new entries but allows reductions).
2. Run `bin/drain_overcap_only.sh --dry-run` to inspect proposed closures.
3. Apply targeted close with buffer ≤3% until utilisation <30%.
4. Review `RISK_ALLOC_TARGET_PCT`, Top-K selection, and live spreads before re-enabling entries.

## Weekend / Market Close
1. Expect WS ages to stretch; allocator-lite keeps previous weights while sessions closed.
2. Leave kill switch ON until the first session opens and freshness checks pass.
3. Resume with the Pre-Open Spin-Up checklist.
