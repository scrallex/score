# Information Systems Mapping Initiative

The live trading stack has evolved into a dense mesh of Python services, shell utilities, TypeScript/React components, and C++ bindings. We now need an authoritative map that explains **where every critical variable lives, how it flows, and which subsystem owns it**. The `docs/diagrams/` workspace is the home for that audit.

## Objectives
- Build a durable catalogue of variables, config knobs, and data channels across `scripts/`, `apps/frontend/`, and `src/`.
- Collapse redundant loops by surfacing when different components carry the same concept under different names.
- Provide diagrams (textual first, then visual) that let us answer “what updates this field?”, “who consumes it?”, and “how does it change live behaviour?”.

## Scope & Cadence
| Phase | Deliverable | Notes |
| --- | --- | --- |
| 1 | **Framework** (this folder) | Define templates, taxonomy, and workflow. |
| 2 | **Variable Catalogue** | Iteratively document variables per subsystem. |
| 3 | **Flow Diagrams** | Translate catalogues into dependency diagrams (Mermaid/PlantUML later). |
| 4 | **Collapse Actions** | Identify & merge duplicate control loops. |

The goal is to move deliberately—accuracy matters more than velocity. Each pass should tighten the map and expose the next highest-risk gap.

## Folder Layout
```
docs/diagrams/
├── README.md                 # Initiative overview (this file)
├── 00_master_index.md        # Master TOC & progress tracker
├── 01_scripts_catalog.md     # Python & shell tooling inventory
├── 02_frontend_catalog.md    # React/Vite variable inventory
├── 03_backend_catalog.md     # Core engine (src/) inventory
├── templates/
│   ├── variable_record.md    # Single-variable template
│   └── system_profile.md     # Subsystem summary template
└── notes/                    # Scratchpad per-audit notes & TODOs
```

## Workflow
1. **Inventory** – use structural scans (`rg`, `pyright`, `ts-node`) to list variable declarations.
2. **Profile** – for each subsystem, fill out `templates/system_profile.md` to capture context (owner, runtime, data sources, sinks).
3. **Catalogue** – document variables in the subsystem catalogue using the `variable_record` template.
4. **Review** – cross-link equivalent variables between subsystems.
5. **Collapse** – propose code changes (separate PRs) to consolidate redundant knobs / data stores.

Each commit should update:
- The relevant subsystem catalogue.
- The master index with coverage metrics (files audited, variables catalogued).
- Notes/TODOs for outstanding work.

## Tooling & Automation Ideas
- `scripts/audit/list_variables.py` (future): parse ASTs to auto-seed catalogues.
- CI guard: ensure catalogues stay up-to-date when new ENV/config vars land.
- Mermaid diagrams exported to `/docs/diagrams/rendered/` once we are ready to visualise.

Let’s use this framework to keep the documentation and the codebase honest.
