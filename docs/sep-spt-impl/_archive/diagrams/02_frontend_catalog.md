# Frontend Variable Catalogue

**Scope**: React/Vite app under `apps/frontend/` (components, contexts, hooks, runtime config, generated API clients).

## 1. System Profile
- **Owner**: UX / Ops integration
- **Runtime**: React 18 + Vite (Node 20+ build, browser runtime)
- **Primary responsibilities**: operator dashboard, status panels, manual controls (kill switch, allocator toggles).
- **External data sources**: REST API (`/api/**`), WebSocket stream (`ws://.../manifold`, etc.), runtime config (`window.__SEP_CONFIG__`).
- **State containers**: React context providers (`WebSocketContext`, `SymbolContext`), localStorage keys, query hooks.

## 2. Variable Mapping Table (TBD)
| Variable | Location | Type | Source | Sinks / Consumers | Notes |
| --- | --- | --- | --- | --- | --- |
| _TBD_ | | | | | |

## 3. Audit Checklist
- [ ] Enumerate runtime config keys produced by `docker-entrypoint.sh` (`API_URL`, `ALLOW_KILL_TOGGLE`, etc.).
- [ ] Document context state shapes (`WebSocketContext`, `SymbolContext`, `AppConfigContext`).
- [ ] Trace kill-switch state flow (REST → header button → trading stance strip).
- [ ] Catalogue chart overlays & feature flags (torches, strands, slices).

## 4. Notes & TODOs
Use `notes/frontend/` for per-component audit notes, screenshots, and outstanding questions.
