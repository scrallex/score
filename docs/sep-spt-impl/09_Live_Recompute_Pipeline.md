# 09. Live Evaluator & Allocator Loop

**Purpose:** Explain how the rolling evaluator and allocator-lite keep the live book aligned with the freshest 30-day quantum metrics, and how to operate/inspect that loop.

## Overview
1. **Candles land** via OANDA stream or candle-fetcher (`scripts/candle_fetcher.py`).
2. **Rolling evaluator** (`scripts/rolling_evaluator.py`) ingests the new data, runs strict-span backtests for each instrument using the current winner params, and updates `bt:rolling:summary:{instrument}`.
3. **Gates blob publish** — evaluator writes `opt:rolling:gates_blob` containing `{ts, gates, cooldowns, bufs}` plus instrumentation counters.
4. **Allocator-lite** (`scripts/allocator_lite.py`) reads the blob every `ALLOC_INTERVAL_SEC`, filters by session/cooldown, ranks eligible instruments, and pushes weights to Valkey `risk:allocation_weights`.
5. **PortfolioManager** consumes the weights, applies exposure caps (30% budget, ~10% per instrument), and reconciles orders while respecting guard thresholds.

## Configuration Knobs
- Evaluator cadence: `EVAL_INTERVAL_SEC`, `GATES_BLOB_MAX_AGE_SEC`, `SESSION_EXIT_MINUTES`, `HYSTERESIS_DEFAULT`, `HYSTERESIS_PAIR_JSON`.
- Allocator scoring: `ALLOC_SCORE_WEIGHT_COHERENCE`, `ALLOC_SCORE_WEIGHT_STABILITY`, `ALLOC_SCORE_WEIGHT_RUPTURE`, `ALLOC_SCORE_WEIGHT_ENTROPY`, optional hazard weight (`ALLOC_SCORE_WEIGHT_HAZARD`) when incorporating λ.
- Allocation cadence & size: `ALLOC_INTERVAL_SEC`, `ALLOC_TOP_K`, `RISK_ALLOC_TARGET_PCT`, `PM_MAX_PER_POS_PCT`, `PM_RAMP_STEP_PCT`.

All knobs above live in `.env.hotband` and are read by the respective services at startup.

## Telemetry & Verification
- **Metrics** (`/metrics` on backend):
  - `sep_rolling_gate{instrument,eligible,buf}` — live eligibility and buffer per instrument.
  - `sep_allocator_selected{instrument}` — Top-K selection with value `1` for active instruments.
  - `sep_allocator_cooldown{instrument}` — cooldown state (1=active cooldown).
  - `sep_span_integrity_ok` — verifies strict-series monotonicity.
- **Valkey keys**:
  - `opt:rolling:gates_blob` — current decision snapshot (`jq` for inspection).
  - `risk:allocation_weights` — allocator-lite output consumed by PortfolioManager.
  - `bt:rolling:summary:{instrument}` — evaluator summaries (Calmar, trades, pnl) per instrument.
- **Logs**:
  - `docker logs sep-rolling-evaluator` (or the evaluator task runner) for gate updates.
  - `docker logs sep-allocator-lite` for publish cadence and staleness warnings.

## Operational Notes
- Allocator-lite holds previous weights when the gates blob is stale or missing; it never improvises.
- Session gating suppresses instruments nearing close (`SESSION_EXIT_MINUTES`) and those outside open hours; `ExitBeforeClose` closes remaining exposure.
- Margin gating halts new entries above `MARGIN_HYST_HIGH`. PortfolioManager still allows reductions.
- Use the **Daily Ops Snapshot** checklist to verify the loop quickly; dig into `/metrics` and Valkey when divergence appears.
- Nightly or weekend automation should rely on the evaluator artifacts instead of ad-hoc cron sweeps; see `program/plan.md` for roadmap actions.
