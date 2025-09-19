# Master Index & Progress Tracker

This index answers two questions:
1. **Where is the latest documentation for each subsystem?**
2. **How complete is the variable mapping audit?**

| Subsystem | Source Roots | Catalogue | Coverage (est.) | Last Audit | Next Actions |
| --- | --- | --- | --- | --- | --- |
| Scripts Tooling | `scripts/`, `bin/` (wrappers) | [01_scripts_catalog](01_scripts_catalog.md) | 70% (warmup, trials, live metrics) | 2025-09-16 | Align hydrator TTL vs repetition gate; evaluate warmup/core loop consolidation (see notes/scripts/deploy_warmup.md). |
| Frontend | `apps/frontend/` | [02_frontend_catalog](02_frontend_catalog.md) | 0% (framework only) | Pending | Inventory global contexts, hooks, env vars. |
| Core Engine | `src/`, `connectors/`, `scripts/backend` glue | [03_backend_catalog](03_backend_catalog.md) | 0% (framework only) | Pending | Map primary services (backend API, allocator-lite, evaluator). |
| Shared Config | `.env.hotband`, `config/` | TBD (link after creation) | 0% | Pending | Decide whether to keep in master doc or per subsystem. |

## Workflow Log
- **2025-09-16** – Live metrics inventory added (manifold C/S/H/R, hazard, gates, weights); live loop diagram + core profile drafted.
- **2025-09-16** – Generated artefact inventory captured (reports, manifolds, snapshots); scripts checklist closed.
- **2025-09-16** – Scripts audit pass: deploy warmup + trials variables documented; diagrams 10/11/12 refreshed; notes logged under `notes/scripts/deploy_warmup.md`.
- **2025-09-16** – Framework created, templates seeded. Awaiting first subsystem audit pass.

## Mermaid Diagrams
- Architecture overview: docs/diagrams/10_architecture_overview.mmd (includes deploy warmup + promotion loop)
- Core data flows: docs/diagrams/11_data_flows.mmd (deploy warmup, trials, Valkey keys)
- Services ↔ variables map: docs/diagrams/12_services_variables.mmd (deploy + ops focus)

## Open Questions
- Preferred diagramming syntax (Mermaid vs PlantUML vs external tool)?
- Should we auto-extract Python globals (`argparse`, env) via scripts or maintain manually?
- How do we represent runtime dynamic keys (Valkey, Redis pub/sub channels) alongside static vars?
- Can warmup loops be collapsed into a single orchestrator without losing observability? (see notes/scripts/deploy_warmup.md)
- Align `ws:last:manifold` TTL with hydrator interval?

Update this index on every audit pass.
