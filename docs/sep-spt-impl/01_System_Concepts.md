1. System Concepts and Implementation (Canonical)
   This is the single source of truth for SEP’s live trading system. The platform is aligned to real‑time quantum metrics — coherence, stability, rupture, entropy — and no longer uses motif infrastructure or the batch optimizer. Allocation is computed directly from metrics; trading admission is gated by guards evaluated over the same metrics.

   See [`05_Core_Trading_Loop.md`](05_Core_Trading_Loop.md) for the concise runtime profile derived from the diagrams templates.

Objective & Constraints (Live)

- Objective: maximize NAV over time.
- Hard constraint: a portfolio budget derived from available margin limits exposure.
  - Portfolio budget: `budget = marginAvailable × RISK_ALLOC_TARGET_PCT` (default 0.30).
  - Per‑instrument cap: `min(budget/K, PM_MAX_PER_POS_PCT×budget)` with ramped increases.
  - K instruments selected/published by allocator‑lite (fill‑to‑K at spin‑up) and consumed by the backend.

Core Concepts

1. System Topology and Separation of Concerns
   Concept: SEP separates metric computation from trading execution and UI. Valkey is the source of truth for manifolds, the signal index, allocation weights, and HUD mirrors. OANDA provides pricing, account, and order execution. The live engine targets a Top‑K portfolio under a 30% target budget and hard kill‑switch.
   Implementation:

Services (`docker-compose.hotband.yml`):

- `sep-backend` (`scripts/trading_service.py`): REST API + StrategyEngine/PortfolioManager.
- `sep-websocket` (`scripts/websocket_service.py`): WebSocket relay for market/manifold/ledger.
- `sep-allocator-lite` (`scripts/allocator_lite.py`): Publishes Top‑K allocation weights to `risk:allocation_weights`.
- `sep-ws-hydrator` (`scripts/ws_hydrator.py`): Mirrors minimal manifold payloads to `ws:last:manifold:*` and keeps coherence fresh.
- `sep-candle-fetcher`: Periodically primes recent candles via the backend.
- `sep-frontend`: Nginx‑served UI.

Note: No separate GPU worker is required. All live metrics are computed inside `sep-backend` (via `libquantum_metrics.so`), and `sep-ws-hydrator` mirrors minimal payloads for the UI. Valkey is internal (`VALKEY_URL=redis://sep-valkey:6379/0`).

2. Quantum Metrics (no motifs)
   Concept: All admission and allocation rely on live manifold-derived metrics:
   - Coherence (C): directional agreement over recent windows
   - Stability (S): persistence of signal structure
   - Rupture (R): structural breaks or instability
   - Entropy (H): uncertainty / noise floor
   Guards enforce C ≥ min, S ≥ min, R ≤ max, H ≤ max. No motif queues, scores, or motif filters are present.

3. Trading Strategy and Execution (Top‑K by Quantum/Price)
   Concept: Track 8 instruments; allocate to Top‑K by a metric score. Maintain used‑margin ≈ 30% of balance with kill‑switch for safety.
   Implementation:
   • Allocator‑Lite (`scripts/allocator_lite.py`):

- Ranks instruments by a weighted quantum score (defaults: C 0.6, S 0.2, R 0.15, H 0.05).
- Publishes ≥K weights every minute (fill‑to‑K); holds prior weights if the consolidated gates blob is stale; publishes a soft fallback only on first boot.
- Publishes normalized weights to `risk:allocation_weights` consumed by the backend.

   • PortfolioManager (`scripts/trading/portfolio_manager.py`):
- Reads `risk:allocation_weights`. Targets per-instrument exposure = weight × budget capped by `min(budget/K, PM_MAX_PER_POS_PCT×budget)`.
- Budget = `marginAvailable × RISK_ALLOC_TARGET_PCT` (default 0.30). Reconciles toward targets continuously; ramps increases; reductions always allowed (provider close API).
- Sizing considers pair-specific margin proxy and risk caps; spread/session/margin gates suppress low‑quality adds.

   • StrategyEngine + GuardEvaluator (`scripts/trading/engine.py`, `scripts/trading/guards.py`):
   - Direction: CombinedDirection uses price slope + coherence delta with admission via a CoherenceGate. Price confirmation is required.
   - Optional exits: slope‑reversal exit (`EXIT_SLOPE_ENABLED=1`) closes positions when slope flips beyond a threshold.
   - PnL thresholds: proportional exit (`EXIT_PNL_ENABLED=1`) closes positions at configurable take‑profit (+0.50) or stop‑loss (−1.25) levels.
   - Guards enforce coherence/rupture/stability/entropy thresholds; spread and VaR gates protect entries.

4. Data Schema (Valkey Keys)

- Signals index: `sep:signal_index:{instrument}` (ZSET of keys)
- Manifolds: `manifold:{instrument}:{YYYY-MM-DD}` (GZIP JSON payload)
- Allocation weights: `risk:allocation_weights` (JSON: { instrument → weight })
- Diagnostics mirrors: `ws:last:manifold:{instrument}`
 - Backtests (studies):
   - Runs (JSON): `bt:trial:run:{run_id}`
   - Trades (JSON list): `bt:trial:trades:{run_id}`
   - Span indices: `bt:trial:index:{instrument}:{hours}`
   - Param‑scoped indices: `bt:trial:index:{instrument}:{hours}:{param_key}` where `param_key = c{c_min}:l{look}:t{thr_bps}:ec{entry_cd}:xc{exit_cd}`


5. Backtest Harness (optional)

- Developer utility: `scripts/backtest_engine.py`. Backtest REST endpoints are not part of the slimmed production API.

6. APIs (selected)

- Health/status: `/health`, `/api/status`
- Pairs, mode: `/api/pairs`, `/api/trading/mode`
- Coherence/readiness: `/api/coherence/status`, `/api/trade/readiness`
- OANDA: `/api/oanda/pricing`, `/api/oanda/account`, `/api/oanda/positions`, `/api/oanda/open-trades` (`/api/trades/open` alias)
- Market data & candles: `/api/market-data`, `/api/candles`, `POST /api/candles/fetch`
- Allocation weights: `/api/risk/allocation-status`
- Coeffs snapshot: `/api/manifold/coeffs`

7. Removed/Deprecated

- Motif endpoints and optimizer/batch endpoints are removed.
- GPU worker has been removed; the stack is CPU‑only.
- Legacy docs `docs/_archive/05_Motif_Intelligence.md` and `docs/_archive/07_Alpha_Generation.md` are retained for historical context; there is no `06_GPU_Operations.md` in the current stack.

8. QFH/QBSA Pipeline (Summary)

Session‑Aware Live Addendum (2025‑09)

- Sessions & cadence
  - Only trades instruments that are open in one of: Tokyo, London, New York, Sydney.
  - Minute‑level evaluation; warmup orchestration primes signal indices and ws-hydrator mirrors `ws:last:manifold:*` for UI freshness.
- Rolling Evaluator V2
  - Recomputes strict 30‑day summaries every `EVAL_INTERVAL_SEC` (default 60s).
  - Adaptive floors + per‑pair hysteresis and cooldown; consolidated blob at `opt:rolling:gates_blob`.
  - Session gating: skip closed; block late adds `≤ SESSION_EXIT_MINUTES` (default 5).
- Allocator‑Lite
  - Reads the consolidated blob; holds publish if blob is stale; filters to open, not‑near‑close instruments.
- Engine & Exits
  - CombinedDirection for direction; `ExitBeforeClose` flattens before session close.
  - Pre‑trade freshness guard; spread and risk checks; OANDA connection verified.
- Pulse Trader (learning cadence)
  - If no trade is planned, seed an entry every `PULSE_INTERVAL_SEC` (~240–300s) on the best open instrument.
  - Sizing respects a dynamic budget and a 30% margin gate (`MARGIN_HYST_HIGH` default 0.30).
- Budget & margin gate
  - Dynamic budget: `max_total_exposure ≈ min(balance, marginAvailable) × RISK_ALLOC_TARGET_PCT (0.30)`; snapshot via `/api/risk/budget`.
  - Gate: hold new entries if `marginUsed/(marginUsed+marginAvailable) ≥ MARGIN_HYST_HIGH`.

   • QBSA encodes candle series into a compact bitstream (direction/range/volume proxies) that feeds QFH.
   • QFH runs sliding‑window analysis producing per‑snapshot metrics (coherence, stability, entropy, rupture) and coefficients (sigma_eff, lambda). It emits a daily manifold JSON.
   • Data model:
     - Candles cache: `md:candles:{instrument}:{gran}` (ZSET, score ts_ms)
     - Signals: `sep:signal:{instrument}:{ts_ns}` with index `sep:signal_index:{instrument}` (ZSET)
     - Manifolds (optional, daily): `manifold:{instrument}:{YYYY-MM-DD}` (gz JSON)
     - WS mirror: `ws:last:manifold:{instrument}` (JSON)
   • Priming & trials:
     - Prime last N days: `python3 scripts/ops/prime_qfh_history.py --days 30`
     - Run trial spec: `python3 scripts/run_backtests.py config/experiments/nightly.yaml`
     - Report best: `python3 scripts/report_trials.py --days 30 --out output/reports/backtests/last_30d_report.md`
   • Native generator: `bin/manifold_generator` accepts `--input` (file or `valkey:{inst}:{from}:{to}`) and outputs file or `valkey`.

Event‑Time Semantics (Backtests)
   • Live trading uses wall‑clock time for engine cooldowns; backtests inject event timestamps (`t_ms`) so cooldowns elapse in historical replay.
   • This ensures the StrategyEngine cadence during backtests matches the timing implied by historical signals.

Span Invariants & Studies
   • Same‑params across spans: 14/30/60 comparisons must use identical `{c_min, look, thr_bps, entry_cd, exit_cd}`.
   • Monotonic series: for a single long run and fixed end, `trades(60) ≥ trades(30) ≥ trades(14)`.
   • API: `GET /api/studies/series?...` returns subwindow counts from a single long run (90d preferred).
   • Selector: `GET /api/backtest/latest-run` accepts param filters and prefers param‑scoped indices to keep spans apples‑to‑apples.
