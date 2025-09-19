# Program Plan & Milestones

Updated: 2025-09-16. The team is operating on the quantum-metrics stack with allocator-lite + rolling evaluator as the single source of truth.

## Workstreams
- **Operations** — runbooks, telemetry, allocator/evaluator health, incident drills.
- **Strategy & Research** — strict span maintenance, exit audits, NAV divergence analysis, gating improvements.
- **Platform & Tooling** — service reliability, CI, automation of warmup/recompute loops, observability.

## 30 / 60 / 90-Day Outcomes
### Day 0–30 (Solidify)
- Complete documentation split (index, checklists, reference) and keep allocator telemetry log current.
- Automate deploy + warmup verification (scripted POST-deploy validator).
- Validate rolling evaluator outputs against manual backtests for two instruments; log results in `docs/reports/`.
- Establish Prometheus alert coverage for gates blob freshness, eligible count, and NAV/Calmar divergence.

### Day 31–60 (Consolidate)
- Integrate allocator-lite score adjustments (λ penalty, session heuristics) and measure impact via controlled trials.
- Add CI smoke tests covering deploy script, allocator-lite health endpoint, and warmup orchestrator entrypoints.
- Stand up Grafana dashboard for allocator and risk metrics using `/metrics` + Valkey snapshots.
- Deliver exit-efficiency audit (MAE/MFE vs realized PnL) with mitigation plan.

### Day 61–90 (Scale & Enhance)
- Launch automated weekend sweep (Prime → Evaluate → Report) driven by rolling evaluator artifacts.
- Run guarded live pilot with documented risk tolerances; compare NAV vs strict Calmar weekly.
- Produce investor-ready documentation pack (architecture profile, telemetry log, live performance summary).
- Prototype multi-broker abstraction or parallelization plan if throughput requires it.

## Role Alignment
- **Quantitative Researcher** — owns strict span quality, gate tuning, exit audits, and NAV divergence investigation.
- **Systems Engineer** — owns service reliability, automation, CI, and telemetry (Prometheus/Grafana, allocator health).
- **Shared** — release readiness (deploy + operations drills), documentation upkeep, incident playbooks.

Progress and reports belong in `docs/reports/` with pointers from `Allocator_Telemetry_Log.md` when data changes.
