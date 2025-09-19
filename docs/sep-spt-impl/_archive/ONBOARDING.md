# New Hire Onboarding (Trading Platform)

Welcome aboard. This guide connects you to the core documentation and gets you productive on the live quantum-metrics stack.

## 0. Start With The Maps
- Read [`docs/index.md`](index.md) to see where each document lives.
- Skim [`operations/checklists.md`](operations/checklists.md) — you will run the Pre-Open Spin-Up checklist on day one.
- Review the architecture duo: [`01_System_Concepts.md`](01_System_Concepts.md) and [`Core_Trading_Loop.md`](Core_Trading_Loop.md).

## 1. Environment Setup (Single Box)
- Platform: Ubuntu droplet (provisioned).
- Internal Valkey: `redis://sep-valkey:6379/0` (internal-only).
- Repo highlights:
  - `scripts/` Python orchestrators and tooling (including warmup orchestrator and rolling evaluator helpers).
  - `apps/frontend/` (React/Vite UI).
  - `src/` C++ quantum metrics kernels.
  - `docs/operations/` for runbooks.

## 2. Bring Up The Stack
1. Run `./deploy.sh` — builds images, starts backend/websocket/allocator-lite/ws-hydrator/candle-fetcher/trade-monitor/frontend, primes data via warmup orchestrator, and keeps the kill switch engaged.
2. Follow the **Pre-Open Spin-Up** checklist to disengage the kill switch safely and verify allocator-lite + rolling evaluator health.
3. Confirm freshness (`ws:last:manifold:*`, `sep:signal_index:*`) and allocator weights on Valkey before allowing live entries.

## 3. Data & Signals (Prime → Verify)
- Warmup orchestrator handles backfill/prime/hydrate steps post-deploy. Re-run a step manually if any freshness check fails:
  - `docker exec -it sep-backend python3 /app/scripts/ops/warmup_orchestrator.py --steps backfill`
  - `docker exec -it sep-backend python3 /app/scripts/ops/warmup_orchestrator.py --steps prime --store-manifold-to-valkey`
  - `docker exec -it sep-backend python3 /app/scripts/ops/warmup_orchestrator.py --steps hydrate`
- Inspect signal coverage per pair (helps validate rolling evaluator inputs):
  ```bash
  docker exec -it sep-backend python3 - <<'PY'
import os, redis
r = redis.from_url(os.getenv('VALKEY_URL'), decode_responses=True)
for p in 'EUR_USD,USD_JPY,GBP_USD,EUR_JPY,USD_CAD,NZD_USD,AUD_USD,USD_CHF'.split(','):
    print(p, r.zcard(f'sep:signal_index:{p}'))
PY
  ```

## 4. Rolling Evaluator & Allocator-Lite
- Rolling evaluator recomputes strict 30-day backtests every `EVAL_INTERVAL_SEC` (default 60s) and writes `opt:rolling:gates_blob`.
- Allocator-lite (`scripts/allocator_lite.py`) reads that blob, filters by session/cooldown, and publishes weights to `risk:allocation_weights` every `ALLOC_INTERVAL_SEC`.
- Monitor `/metrics` for `sep_rolling_gate` and `sep_allocator_selected`; investigate if eligibles <3 for more than 10 minutes.

## 5. Kill-Switch Discipline & Risk Checks
- Keep kill switch ON until the Pre-Open Spin-Up checklist passes.
- Inspect `GET /api/risk/allocation-status` for target exposure and `GET /api/risk/budget` for dynamic budget.
- Use `bin/drain_overcap_only.sh --dry-run` before lowering exposure with kill switch engaged.

## 6. First Week Objectives
1. Run deploy + spin-up end-to-end under supervision; document any friction in `docs/reports/`.
2. Validate allocator-lite/rolling evaluator telemetry in Prometheus (`sep_rolling_gate`, `sep_allocator_selected`, `sep_span_integrity_ok`).
3. Reproduce a backtest via `scripts/backtest_engine.py` for one instrument using strict params (align with rolling evaluator winners).
4. Draft a one-page observation log: gating behaviour, NAV vs Calmar alignment, and proposed improvements.

For roadmap, milestones, and role expectations see [`program/plan.md`](program/plan.md).
