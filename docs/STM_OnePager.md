# Structural Precursors in MMS Telemetry (STM Portable Tool)

## What We Proved
- **Structural twins:** 3 cross-day storm→storm twins, each with 50 aligned windows and mean ANN distance ≈ 2×10⁻³.
- **Early-warning trend:** Foreground density rises to **7.4%** in the final 5 minutes before onset while the quiet baseline stays flat.
- **Guardrailed reproducibility:** Router auto-tunes percentiles to keep foreground at 5–7%; every figure/table links to JSON/CSV outputs and rebuilds with `make all`.

## Why It Matters
- Gives ops an interpretable early-warning layer that complements existing thresholds/alarms.
- Pulls prior episodes instantly (structural twins + evidence) to accelerate triage and playbook recall.
- Portable across missions/logs thanks to adapters and percentile guardrails.

## Ready-To-Run Components
- `stm` CLI (ingest, report, stream, onsets autolabel).
- Adapters: MMS, THEMIS, generic CSV (bit features: `__UP`, `__ACCEL`, `__RANGEEXP`, `__ZPOS`).
- Streaming runtime (`stm stream --config configs/mms_stream.yaml`) → `/stm/health`, `/stm/seen`.
- Scripts + Makefile: `make all` regenerates scorecard, plots, lead-times, diagnostics.

## Pilot Offer (4–8 Weeks)
1. Run STM on 1–2 of your datasets + quiet baseline.
2. Deliver scorecard (coverage, twins, ANN, lead-time trend) + streaming demo (optional).
3. Provide reproducibility bundle, docs, and next-step roadmap.

## Contact
- Snapshot repo package: `MMS_structural_precursors_v0.1.zip`
- QuickStart & API docs: `/docs/TOOL_QUICKSTART.md`, `/docs/API.md`
- Reach: <your-name@company.com> / <phone>
