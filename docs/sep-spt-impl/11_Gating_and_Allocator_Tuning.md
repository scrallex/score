# Gating and Allocator Tuning

This document captures the minimal configuration that keeps the live echo-driven stack aligned. All gating is now based on the repetition count and hazard emitted by the C++ manifold builder.

## Echo Gate
- A signal is **eligible** when:
  - `repetition_count_1h >= ECHO_MIN_REPETITIONS` (default `3`)
  - `lambda_hazard <= ECHO_HAZARD_MAX` (default `0.70`)
- Environment defaults:
  ```bash
  ECHO_MIN_REPETITIONS=3
  ECHO_HAZARD_MAX=0.70
  ECHO_LOOKBACK_MINUTES=60   # informational; repetition window baked into C++
  ECHO_SIGNATURE_PRECISION=2
  ```
- No hysteresis / cooldowns remain. The evaluator publishes a gate each cycle based purely on the current fingerprint.

## Gates Blob Contract
The evaluator writes a consolidated blob after checking every instrument:

```json
{
  "ts": 1758059154076,
  "gates": {"EUR_USD": 1, ...},
  "repetition": {"EUR_USD": {"count_1h": 4, "first_seen_ms": 1758058120000}},
  "hazard": {"EUR_USD": 0.12, ...},
  "bufs": {"EUR_USD": 0},
  "cooldowns": {"EUR_USD": 0}
}
```

- Valkey key: `opt:rolling:gates_blob`
- Allocator-lite only reads this blob; if missing or stale it holds previous weights and logs a warning.

## Allocator-lite Touchpoints
- `ALLOC_INTERVAL_SEC=60`
- `ALLOC_TOP_K=3`
- Weights are published only for eligibles; when no instruments satisfy the gate the previous weights are kept.

## Telemetry
- `sep_rolling_gate{instrument="...", eligible="1"}` — primary Prometheus signal.
- `sep_allocator_selected{instrument="..."}` — allocator output for Top-K instruments.
- Allocator status endpoint `curl :8100/status | jq` returns cadence and count of eligibles per cycle.

## Session & Margin Guards
- Session gating: `SESSION_TRADING_ENABLED=1`, `SESSION_EXIT_MINUTES=5` — evaluator suppresses instruments outside open sessions or near close.
- Margin gating: PortfolioManager halts new entries when utilisation ≥ `MARGIN_HYST_HIGH` (default 0.30). Budget = `marginAvailable × RISK_ALLOC_TARGET_PCT`.

## Required Environment Summary
```
ECHO_MIN_REPETITIONS=3
ECHO_HAZARD_MAX=0.70
SESSION_TRADING_ENABLED=1
SESSION_EXIT_MINUTES=5
ALLOC_INTERVAL_SEC=60
ALLOC_TOP_K=3
RISK_ALLOC_TARGET_PCT=0.30
MARGIN_HYST_HIGH=0.30
```

Keep this document in sync with the evaluator and allocator code when thresholds change.
