# Operations Reference

Complement to `operations/checklists.md`. Each section explains the guardrails, configuration knobs, and tooling that underpin the live quantum-metrics stack.

## Guardrails & Defaults
- **Strict spans** — selection derives from a single long run; `sep_span_integrity_ok` must stay `1`.
- **Adaptive gates** — base floors: `min_trades=10`, `calmar=0.15`; hysteresis buffer default `0.10`, clamped `[0.00,0.30]` with pair overrides via `HYSTERESIS_PAIR_JSON`.
- **Cooldown** — 4 consecutive misses trigger a 2h cooldown (`opt:rolling:cooldown:{instrument}`) and reset pass/fail counters.
- **Exposure cap** — total allocation ≤30% of margin available; per instrument cap `min(budget/K, PM_MAX_PER_POS_PCT × budget)`; ramp increases limited by `PM_RAMP_STEP_PCT`.
- **Live guard thresholds** — `GUARD_MIN_COHERENCE ≥ 0.55`, `AUTO_MAX_RUPTURE`, `AUTO_MAX_ENTROPY`, `AUTO_MIN_STABILITY` enforced inside PortfolioManager/StrategyEngine.

## Rolling Evaluator & Allocator-Lite
- **Gates blob** — `opt:rolling:gates_blob` holds `{ts, gates, cooldowns, bufs}`. Allocator-lite only trusts this blob; if stale beyond `GATES_BLOB_MAX_AGE_SEC` it retains prior weights and logs a warning.
- **Metrics** — `/metrics` exposes `sep_rolling_gate{instrument,eligible,buf}`, `sep_allocator_selected{instrument}`, `sep_allocator_cooldown{instrument}`, `sep_span_integrity_ok`.
- **Session gating** — evaluator skips closed markets and flips instruments to ineligible within `SESSION_EXIT_MINUTES` (default 5) of session close. `ExitBeforeClose` flattens residual positions.
- **Margin gating** — entries pause when `marginUsed / (marginUsed + marginAvailable) ≥ MARGIN_HYST_HIGH` (default 0.30). Hysteresis window defined by `MARGIN_HYST_LOW`/`HIGH`.

## Key API Endpoints
- Health: `GET /health`, `GET /api/status`.
- Instruments & sessions: `GET /api/pairs`, `GET /api/ranking/active` (Top-K context with metrics & session metadata).
- Risk & allocation: `GET /api/risk/allocation-status`, `GET /api/risk/budget`, `POST /api/trade/reconcile` (manual sync), `POST /api/kill-switch`.
- Manifolds & coherence: `GET /api/manifold/coeffs?instrument=...`, `GET /api/coherence/status?instrument=...`.
- Market data: `GET /api/candles`, `POST /api/candles/fetch` (candle-fetcher handles cadence).
- OANDA integration: `GET /api/oanda/account`, `/api/oanda/positions`, `/api/oanda/open-trades`, `/api/oanda/instruments`.

## Tooling & Scripts
- **deploy.sh** — builds containers, primes data via warmup orchestrator, and restarts backend/websocket with live config.
- **Warmup orchestrator** — `scripts/ops/warmup_orchestrator.py` handles `backfill`, `prime`, and `hydrate` steps after deploy.
- **Drain helper** — `bin/drain_overcap_only.sh` closes overweight exposure while kill switch engaged; supports `--dry-run`, `--buffer-pct`, `--min-units` overrides.
- **Logs** — `docker logs sep-backend | egrep -i 'StrategyEngine|GuardEvaluator|PortfolioManager'` for trade gating visibility.

## Telemetry & Alerts
- Prometheus rules (group `sep-rolling`) guard blob freshness, eligible counts, and cooldown lockouts.
- Watch NAV divergence: investigate exits/slippage if NAV declines while selected Calmar remains positive (`avg_over_time(sep_bt_calmar{instrument=selected}[6h]) > 0`).
- Expect lower WS activity on weekends/market close; allocator-lite keeps previous weights but exposes stale-age metrics.

## Environment Knobs (operations-facing)
- Allocation: `ALLOC_TOP_K`, `RISK_ALLOC_TARGET_PCT`, `PM_MAX_PER_POS_PCT`, `PM_RAMP_STEP_PCT`.
- Freshness: `FRESH_MAX_SIGNAL_AGE_SEC`, `FRESH_MAX_WS_AGE_SEC` (warmup orchestrator ensures coverage post-deploy).
- Evaluator cadence: `EVAL_INTERVAL_SEC`, `GATES_BLOB_MAX_AGE_SEC`, `SESSION_EXIT_MINUTES`.
- Safety: `KILL_SWITCH` (backed by Valkey `ops:kill_switch`), `READ_ONLY`, `MAX_SPREAD_BPS`, `MARGIN_HYST_LOW/HIGH`.
