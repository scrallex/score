# Initial Audit Notes (2025-09-16)

## Primary Control Loops to Collapse
1. **Deployment Warmup Loop** – `deploy.sh` → backfill → prime → hydrate. Need to map env flags (`BACKFILL_ON_DEPLOY`, `RUN_TRIALS_ON_DEPLOY`) and where they echo in backend containers.
2. **Gating / Allocation Loop** – rolling evaluator publishes `opt:rolling:gates_blob`, allocator-lite consumes env + Valkey + best configs. Variables: `AUTO_MIN_COHERENCE`, `ALLOC_TOP_K`.
3. **Frontend State Loop** – kill-switch, stance strip, diagnostics. Variables: `window.__SEP_CONFIG__` keys, REST responses, WS events.

## Immediate TODOs
- Scripts: enumerate env vars in `deploy.sh` (use `rg 'export ' deploy.sh`).
- Frontend: document `WebSocketContext` shape and all localStorage keys.
- Backend: trace `scripts/allocator_lite.py` arg parser + env defaults.

## Tooling Ideas
- Python AST scan to capture module-level `os.getenv` usages.
- TypeScript codemod to list `process.env` (during build) and `window.__SEP_CONFIG__` references.
- Valkey key introspection script (list keys by prefix, attach producer/consumer).
