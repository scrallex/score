# Backend / Core Engine Variable Catalogue

**Scope**: `src/` (Python backend services, API, allocator-lite), `connectors/`, `apps/backend` if present, plus C++ bindings (`bin/libquantum_metrics.so`) where they expose configuration knobs.

## 1. System Profile
- **Owner**: Quant Engineering
- **Runtime**: Python 3.12 (Flask/FastAPI backend), auxiliary workers (allocator-lite, rolling evaluator), compiled C++ libs.
- **Primary responsibilities**: market data ingestion, gating/evaluation, allocation publishing, REST/WS APIs, Valkey orchestration.
- **External dependencies**: OANDA streaming/rest, Valkey, Docker-managed services, file outputs under `output/`.
- **Key configs**: `.env.hotband`, `config/*.yaml`, `opt:best_config:*` (Valkey), environment variables injected via compose.

## 2. Variable Mapping Table (TBD)
| Variable | Location | Type | Source | Sinks / Consumers | Notes |
| --- | --- | --- | --- | --- | --- |
| _TBD_ | | | | | |

## 3. Audit Checklist
- [ ] Document REST endpoints and their config/env dependencies.
- [ ] Trace allocator-lite configuration (env vars → Valkey reads → published weights).
- [ ] Map rolling evaluator thresholds (guardrails, hysteresis) and their env/env overrides.
- [ ] Capture Valkey channel usage (`ws:manifold`, `opt:rolling:gates_blob`, etc.).

## 4. Notes & TODOs
Use `notes/backend/` for in-progress diagrams and unresolved questions.
