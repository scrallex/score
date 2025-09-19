## System Profile: Core Trading Loop
- **Owner / Point-of-contact**: Trading Ops & Quant Engineering
- **Code Roots**: `scripts/ops/warmup_orchestrator.py`, `scripts/rolling_evaluator.py`, `scripts/allocator_lite.py`, `scripts/trading_service.py`, `scripts/ws_hydrator.py`, `src/core/qfh.*`, `bin/manifold_generator`
- **Runtime**: Docker services — `sep-candle-fetcher`, `sep-ws-hydrator`, `sep-rolling-evaluator` (task runner), `sep-allocator-lite`, `sep-backend` (StrategyEngine + PortfolioManager), `sep-websocket`, `sep-frontend`
- **Responsibilities**:
  - Encode OANDA candles into QFH manifolds (coherence, stability, rupture, entropy, λ).
  - Mirror live manifold snapshots for UI/API consumption and ensure freshness.
  - Run strict-span rolling backtests every candle, update adaptive gate thresholds, and publish the consolidated gates blob.
  - Rank eligible instruments and publish Top-K portfolio weights under the 30% exposure budget.
  - Enforce guard thresholds, kill-switch, and margin caps while reconciling live orders.

- **Inputs**:
  - **Environment variables**: `HOTBAND_PAIRS`, `EVAL_INTERVAL_SEC`, `GATES_BLOB_MAX_AGE_SEC`, `EVAL_MIN_TRADES`, `EVAL_CALMAR_MIN`, `EVAL_PNL_MIN`, `HYSTERESIS_DEFAULT`, `HYSTERESIS_PAIR_JSON`, `SESSION_TRADING_ENABLED`, `SESSION_EXIT_MINUTES`, `ALLOC_INTERVAL_SEC`, `ALLOC_TOP_K`, `ALLOC_SCORE_WEIGHT_COHERENCE`, `ALLOC_SCORE_WEIGHT_STABILITY`, `ALLOC_SCORE_WEIGHT_RUPTURE`, `ALLOC_SCORE_WEIGHT_ENTROPY`, `ALLOC_SCORE_WEIGHT_HAZARD`, `RISK_ALLOC_TARGET_PCT`, `PM_MAX_PER_POS_PCT`, `PM_RAMP_STEP_PCT`, `AUTO_MIN_COHERENCE`, `GUARD_MIN_STABILITY`, `AUTO_MAX_RUPTURE`, `AUTO_MAX_ENTROPY`, `MARGIN_HYST_LOW`, `MARGIN_HYST_HIGH`.
  - **Valkey keys / streams**: `md:candles:{instrument}:M1`, `sep:signal_index:{instrument}`, `sep:signal:{instrument}:{ts_ns}`, `ws:manifold` pub/sub, `opt:rolling:gates_blob`, `bt:rolling:summary:{instrument}`, `risk:allocation_weights`, `ops:kill_switch`.
  - **REST / WS endpoints**: `/api/candles/fetch`, `/api/coherence/status`, `/api/manifold/coeffs`, `/api/oanda/*`, `/api/kill-switch`, `/api/risk/allocation-status`.

- **Outputs**:
  - **Valkey**: `ws:last:manifold:{instrument}`, `bt:rolling:summary:{instrument}`, `opt:rolling:gates_blob`, `sep_rolling_gate` exporter cache, `risk:allocation_weights`, `risk:alloc_budget`.
  - **REST / WS**: `/metrics` (Prometheus export), `/status` endpoints for allocator-lite and hydrator, WebSocket feed for manifold updates, OANDA order management via backend API.
  - **Files**: Optional manifold snapshots in `output/manifolds/{instrument}/`, reports in `docs/reports/` when exported via tooling.

- **Critical Variables**:
  - `AUTO_MIN_COHERENCE`, `GUARD_MIN_STABILITY`, `AUTO_MAX_RUPTURE`, `AUTO_MAX_ENTROPY` — live guard thresholds enforced before entry.
  - `HYSTERESIS_DEFAULT`, `HYSTERESIS_PAIR_JSON` — buffer percentages controlling eligibility flips per instrument.
  - `EVAL_MIN_TRADES`, `EVAL_CALMAR_MIN`, `EVAL_PNL_MIN` — adaptive gate baselines for strict-span summaries.
  - `ALLOC_TOP_K`, `ALLOC_SCORE_WEIGHT_*`, `RISK_ALLOC_TARGET_PCT`, `PM_MAX_PER_POS_PCT`, `PM_RAMP_STEP_PCT` — allocation distribution and ramp control.
  - `MARGIN_HYST_LOW`, `MARGIN_HYST_HIGH` — margin utilisation hysteresis for entry gating.

- **Failure Signals**:
  - `ws:last:manifold:{instrument}` stale (> `FRESH_MAX_WS_AGE_SEC`) or missing coherence values.
  - `opt:rolling:gates_blob.ts` older than `GATES_BLOB_MAX_AGE_SEC`.
  - `sep_allocator_selected{}` count deviates from `ALLOC_TOP_K` or `risk:allocation_weights` stops updating.
  - Prometheus alerts: eligible count <3 for 10 minutes, cooldown lockout ≥5 instruments for 30 minutes, NAV down 7d while strict Calmar positive.
  - OANDA API rejections or PortfolioManager errors during reconcile loop.

- **Dependencies**:
  - OANDA REST + pricing stream, Valkey, `libquantum_metrics.so`, Docker runtime, warmup orchestrator for deploy-time data seeding, Prometheus/Grafana for telemetry.

- **Open Risks / TODOs**:
  - Finalize λ-weight integration in allocator scoring and validate against live performance.
  - Extend automated post-deploy validator to cover evaluator freshness and allocator publish cadence.
  - Improve exit-efficiency telemetry (MAE/MFE capture) to feed guardrail adjustments.
