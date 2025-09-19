# Structural Twin Manifold Pilot – Statement of Work (Template)

## Scope
- Analyse customer telemetry/logs (1–2 datasets + quiet baseline) using STM.
- Produce scorecard (foreground coverage, twins ≥ N windows, mean ANN distance, lead-time vs quiet).
- Optional: deploy streaming runtime (`/stm/seen`) for a subset of live data.

## Timeline (4–8 Weeks)
1. **Week 1:** Data handoff, adapter alignment, QuickStart walkthrough.
2. **Weeks 2–3:** Calibration, scorecard generation, onset tuning.
3. **Week 4:** Streaming/alert demo (optional), deliverables handoff, readout.

## Deliverables
- `make all` reproducibility bundle (states, configs, CSV/PNG outputs).
- Scorecard CSV + figure pack + diagnostics JSONs.
- Streaming runtime container/Docker Compose (optional).
- Final report with findings, lead-time trends, recommended next steps.

## Customer Responsibilities
- Provide sample data (CSV/HDF5/CDF) with timestamps & channel schema.
- Supply known incident/onset markers or accept STM autolabel rules.
- Assign SME (~1 hr/week) for domain validation.

## Success Criteria
- ≥ 2 structural twins with ≥ N aligned windows (default N=30) and mean ANN ≤ 3×10⁻³.
- Lead-time final-bin density exceeds quiet baseline by agreed delta (e.g., +3σ).
- Guardrailed foreground within 5–20% for each slice.

## Pricing (Reference)
- Batch pilot: $50–150k (scope-dependent).
- Streaming add-on: +$25–50k.
- Production support/expansion: custom.

## Terms
- On-prem or customer cloud; no external data egress.
- Customer retains data; STM team retains tooling IP.
- Follow-up production SOW available upon successful pilot.

> Customize the numbers/dates before sending to a prospect.
