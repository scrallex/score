4.  Developer Guide
    This guide defines patterns, conventions, and the API contract for engineers contributing to the SEP codebase.
    For a runtime contract of the engine, see [`05_Core_Trading_Loop.md`](05_Core_Trading_Loop.md) alongside the detailed sections below.
    It consolidates frontend architecture, contributing processes, and verification checklists. All changes must align with the quantum‑metrics system documented in `01_System_Concepts.md`. Manual fetch is forbidden; use the typed generated client exclusively.
    Stack Overview

Framework: React 18 + Vite 5 + TypeScript.
Routing: react-router-dom v6.
Charts: Recharts for plots/sparklines.
API: Typed generated client in src/services/apiTyped.ts (from OpenAPI); hooks like useApi, useMutation, usePolling in src/hooks/.
WebSocket: WebSocketProvider in src/context/WebSocketContext.tsx.
Build: Vite with manual chunking (vendor, recharts, dashboard, studies) in vite.config.ts.
Security: CSP blocks inline scripts; external scripts only in public/js/.

Project Layout

src/components/ — Reusable UI panels and widgets.
src/pages/ — Route-level views (Dashboard, Studies, Health, Settings).
src/context/ — React contexts (Symbol, WebSocket).
src/hooks/ — Custom hooks (useApi, useMutation, usePolling).
src/services/apiTyped.ts — Generated typed API client; raw fetch prohibited.
src/styles/tokens.ts — Design tokens for colors, borders, text.
src/types/api.d.ts — Generated types from OpenAPI.

Backend Strategy and Portfolio (Engineering Notes)

- Tunables (env): `ALLOC_TOP_K`, `RISK_ALLOC_TARGET_PCT`, guard thresholds (`GUARD_MIN_COHERENCE`, `GUARD_MIN_STABILITY`, `GUARD_MAX_RUPTURE`, `AUTO_MAX_ENTROPY`).
- Writes: `risk:allocation_weights` (authoritative portfolio weights).
  • PortfolioManager (`scripts/trading/portfolio_manager.py`):
- Reads allocation weights; sizes per dynamic budget; reconciles ~every 30s; honors readiness/guards.
- Budget: balance × `RISK_ALLOC_TARGET_PCT` (persisted to RiskManager limits).
- Units: computed via a fixed internal inverse‑leverage proxy; portfolio budget controls total exposure.
  • StrategyEngine + GuardEvaluator:
- Entry direction via CombinedDirection (price slope + coherence delta) subject to guard thresholds and margin gating.
- TP/SL: Attached on fill. ATR mode via `AUTO_TP_MODE=atr`, `AUTO_TP_ATR_GRAN`, `AUTO_TP_ATR_N`, `AUTO_TP_ATR_K`, `AUTO_TP_RR`. Fixed fallback uses `AUTO_SL_PIPS`.

Runtime Config Surfaces (env)
• Core sizing: `RISK_ALLOC_TARGET_PCT` (budget = marginAvailable × pct)
• Strategy thresholds: `MARGIN_AWARE_THRESHOLD`, `MARGIN_OVERLOAD`, guard thresholds above
• Brackets: `AUTO_TP_ON_FILL_ENABLED`, `AUTO_TP_MODE`, `AUTO_TP_ATR_GRAN`, `AUTO_TP_ATR_N`, `AUTO_TP_ATR_K`, `AUTO_TP_RR`, `AUTO_SL_PIPS`
• Allocator: `ALLOC_TOP_K`
• Safety: `KILL_SWITCH` (Valkey key `ops:kill_switch` is the runtime source of truth)
• Kill switch plumbing (scripts/trading/api_control.py, http_api): `/api/kill-switch` toggles env‑backed flag; entries are gated when engaged; closes always allowed.

Environment Variables Index (Core)

- Risk & Allocation: `RISK_ALLOC_TARGET_PCT`, `RISK_MAX_POSITION_SIZE`, `RISK_MAX_DAILY_LOSS`, `RISK_MAX_POSITIONS_PER_PAIR`, `RISK_MAX_TOTAL_POSITIONS`, `RISK_MAX_NET_UNITS_PER_PAIR`.
- Guard Thresholds: `GUARD_MIN_COHERENCE`, `GUARD_MIN_STABILITY`, `GUARD_MAX_RUPTURE`, `AUTO_MAX_ENTROPY`.
- Direction/Signals: `AUTO_DIRECTION_MODE` (default combined), `AUTO_DIRECTION_LOOKBACK`, `AUTO_DIR_THRESHOLD_BPS`.
- Brackets (TP/SL): `AUTO_TP_ON_FILL_ENABLED`, `AUTO_TP_MODE` (fixed|atr), `AUTO_TP_ATR_GRAN`, `AUTO_TP_ATR_N`, `AUTO_TP_ATR_K`, `AUTO_TP_RR`, `AUTO_SL_PIPS`.
- Safety & Ops: `KILL_SWITCH`, `READ_ONLY`, `LIVE_TRADING_ENABLED`, `MAX_SPREAD_BPS`.
- Data & Store: `VALKEY_URL` (internal), `HOTBAND_PAIRS`, `HOTBAND_WINDOW_DAYS`.
- Services: `API_PORT`, `WS_PORT`, `LOG_LEVEL`, `QUANTUM_METRICS_LIB`.
- WebSocket: `WS_REQUIRE_DIAGNOSTICS`, `WS_REQUIRE_NATIVE` (authenticity gating), `WS_MAX_QUEUE`.

See `scripts/trading_service.py` and `scripts/trading/*` for defaults and parsing; runtime choices persist to Valkey where applicable (`ops:kill_switch`, `risk:allocation_weights`).

API Patterns and Contract
The API contract is defined by `apps/frontend/public/static/sep.openapi.yaml`, audited against routes in `scripts/trading_service.py`. Regenerate types after updates.
Typed Client Only: Use `src/services/apiTyped.ts` for all communication. Manual fetch is deprecated.

Studies API (strict spans)
- `GET /api/studies/default-window` → `{ days, hours }`
- `GET /api/backtest/latest-run?instrument=...&days=...&c_min=...&look=...&thr_bps=...&entry_cd=...&exit_cd=...&min_trades=0`
  - Selects the best (max‑trades) run for the exact params; falls back to generic span only if no param match exists.
- `GET /api/backtest/trades?run_id=...` → `{ run_id, trades: [...] }`
- `GET /api/backtest/results?run_id=...` → persisted run summary
- `GET /api/studies/series?instrument=...&c_min=...&look=...&thr_bps=...&entry_cd=...&exit_cd=...`
  - Returns `[ {days:14, trades}, {days:30, trades}, {days:60, trades} ]` from a single long run (monotonic by design).
- `POST /api/studies/enqueue-long-run` → `{ queued: <key> }` to queue a strict 90‑day run for missing series.

Persistence & Indices
- Runs: `bt:trial:run:{run_id}` (30d TTL)
- Trades: `bt:trial:trades:{run_id}` (30d TTL)
- Span index: `bt:trial:index:{instrument}:{hours}`
- Param index: `bt:trial:index:{instrument}:{hours}:{param_key}` where `param_key = c{c_min}:l{look}:t{thr_bps}:ec{entry_cd}:xc{exit_cd}`

Telemetry (Prometheus)
- `sep_span_integrity_ok` gauge (1/0)
- `sep_studies_series_calls_total{status}` counters (ok, no_long_run, error)
- `sep_span_monotonic_violations_total` counter

GET (useApi): import a typed function from the generated client and pass parameters via the hook. Example: fetch allocation status or coherence status for an instrument.

POST (useMutation): use typed mutation helpers for actions like toggling trading mode or kill switch.

Polling: Use `usePolling` for live data (e.g., allocation status, readiness). `VITE_API_URL` defaults to same‑origin. Typical polling interval: 5000ms.

Build and Dev

Dev: `npm -C apps/frontend run dev` (proxies `/api` and `/ws`).
Build: `npm -C apps/frontend run build`.
Preview Docs: `mkdocs serve`.

CSP and External Scripts

Inline scripts blocked; use public/js/ for helpers.

Verification Checklist

- Trading: Mode toggle `POST /api/trading/mode` and kill switch endpoints respond.
- Allocation: `GET /api/risk/allocation-status` returns weights; only Top‑K are non‑zero.
- Readiness: `GET /api/trade/readiness` shows coherence/guard state per instrument.
- Data: `/api/oanda/pricing`, `/api/candles`, `/api/market-data` respond and match UI.

Backtest Harness (optional)
- Developer utility: see `scripts/backtest_engine.py` and studies under `studies/` for research collection.

For operations, see `02_Operations_Runbook.md`.

Testing Strategy

- Unit (fast): RiskManager gates (caps, ceilings), price formatting precision, ATR bracket math, health route shapes.
- Service (mocked I/O): HTTP handlers with dummy TradingService, OANDA client under `responses`/`requests_mock` to assert retry/backoff and error handling.
- Integration (hermetic): Spin Valkey, assert ws‑hydrator and allocator‑lite write expected keys (`risk:allocation_weights`, `ws:last:manifold:*`); backend boot smoke (`/health`, `/api/pairs`, `/api/candles`) with `READ_ONLY=1`, `LIVE_TRADING_ENABLED=0`.
- End‑to‑End (optional): Compose up backend+websocket+sidecars and probe health; label to skip in CI.

Safety defaults for tests: `READ_ONLY=1`, `DRY_RUN_TRADES=1`, `LIVE_TRADING_ENABLED=0`. Prefer Python 3.12; use pytest with `pytest-asyncio` where needed.

Span Invariant Tests (add in CI)
- Series monotonicity: slice a synthetic long run and assert `14 ≤ 30 ≤ 60`.
- Strict selector: prefer param‑scoped index when params provided; fallback selects max‑trades.
- No pre‑filter on persist: inject a low‑count run; it must be present in storage.
- Cooldown scope: per `(strategy_name, instrument)`; no cross‑pair bleed.

Backtest Protocol (Event-Time Cadence)

- The StrategyEngine honors cooldowns using event-time when `MarketContext.extras['t_ms']` is provided; live uses wall-clock.
- The harness replays historical signals and injects `t_ms` so cooldowns elapse correctly in backtests.
- YAML specs for sweeps (`scripts/run_backtests.py`):
  - Encode overrides under `instruments[].overrides` with scalars or arrays for grid sweeps.
  - `window_days` controls history span per run; omit `hours` unless you need a shorter window.
  - Include `valkey.enabled: true` to persist results for `report_trials.py`/UI consumers.
- Examples:
  - Single run: `python3 -c "from scripts.backtest_engine import run_backtest; print(run_backtest('EUR_USD', 30, overrides={'c_min':0.55,'direction_lookback':12,'dir_threshold_bps':2.0,'entry_cooldown_sec':60,'exit_cooldown_sec':60}, persist=False))"`
  - Grid: `python3 scripts/run_backtests.py config/experiments/nightly.yaml`
- Reporting: `python3 scripts/report_trials.py --days 30 --min-trades 5 --out output/reports/backtests/last_30d_report.md` writes winners to `opt:best_config:*` and persists a Markdown report.

Frontend Runtime & WS Types

- Runtime Config (`window.__SEP_CONFIG__`): Typed via `src/types/runtime.ts` and declared in `src/types/global.d.ts`. Read through `readRuntimeConfig()`; keys include `API_URL`, `API_BEARER_TOKEN`, `WS_URL`, `WS_TOKEN`, `READ_ONLY`, `ALLOW_KILL_TOGGLE`.
- API Client: `src/services/apiTyped.ts` reads runtime config, exports `api` and the resolved `apiBaseUrl`.
- WebSocket Models: `src/types/ws.ts` defines `Candle`, `ManifoldEvent`, `WsMetrics`, `SubscribeMsg` plus helpers `isRecord()` and `toMs()`.
- WebSocket Context: `src/context/WebSocketContext.tsx` now:
  - Parses frames with typed guards (market/manifold/ledger/system).
  - Normalizes candles from WS/REST with a single mapper and updates `marketData: Record<string, Candle[]>`.
  - Exposes `subscribe(channels, instruments?)` using `SubscribeMsg`.
 - Chart Candles: `src/components/OandaCandleChart.tsx` introduces `ChartCandle` and removes most `any` in candles→shapes paths. It enriches chart data with nearest manifold metrics (coherence/stability/entropy/rupture) using typed `ManifoldEvent` accessors.
 - Recharts Internals: `Customized` overlay and `Tooltip` now use minimal local interfaces for width/height and payload items to avoid `any` while keeping compatibility with Recharts’ loose types.
- Diagnostics: `src/components/SystemEndpoints.tsx` shows API/WS endpoints and `wsMetrics.avg_rtt_ms` for ops.

Dev Proxy & Nginx

- Vite Proxy: Only absolute `VITE_API_PROXY`/`VITE_WS_PROXY` are honored; relative `/api` is ignored for proxy target selection. Defaults map to `http://localhost:8000` and `ws://localhost:8001` in dev.
- Nginx Templates: Container renders from `nginx.tmpl.conf` or `nginx.local.tmpl.conf` using env (`SERVER_NAME`, `BACKEND_UPSTREAM`, `WS_UPSTREAM`, `CERT_BASE`, `USE_LOCAL_NGINX`). See `apps/frontend/docker-entrypoint.sh`.
