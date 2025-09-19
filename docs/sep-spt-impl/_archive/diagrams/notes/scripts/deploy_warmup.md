# Deploy Warmup Audit — 2025-09-16

## Observations
- Deploy pipeline runs three warmup loops (candles, signals, manifolds) that all read `HOTBAND_PAIRS`, `CANDLE_GRAN`, and `CANDLE_COUNT`.
- Environment defaults live in `.env.hotband`, but each Python helper still embeds its own fallback list → risk of drift.
- Trials promotion writes both YAML (`config/live_params.yaml`) and Valkey (`opt:best_config:*`); deploy orchestrates both.

## Collapse Candidates
1. Wrap warmup helpers into a single `scripts/ops/warmup.py` invoked once from `deploy.sh` to avoid repeating API fan-out.
2. Centralise the hotband pair list via `scripts.shared_utils.config_loader` so CLI defaults and env stay aligned.

## Follow-ups
- ✅ Documented artefacts from `report_trials.py`, `capture_live_snapshot.py`, and warmup outputs (see `docs/diagrams/01_scripts_catalog.md`).
- Verify `ws:last:manifold` TTL matches websocket hydrator cadence (currently 300s vs hydrating every 10s).

- 2025-09-16: Created `scripts/ops/warmup_orchestrator.py` to consolidate backfill/prime/hydrate steps; deploy.sh now delegates to this entrypoint.
