# Every line must carry its receipts — Operational Roadmap

## Core Demo Pillars
- **Truth-pack ingestion**: curate 80–150 page corpus → build target manifold with QFH metrics, repetition counts, semantic centroid seeds.
- **Reality filter panels**: naïve semantic, naïve structural, hybrid reality filter sharing the same stream.
- **KPIs**: hallucination rate, repair yield, citation coverage, latency per line (ms).
- **Evidence bundle**: every approved span carries twin IDs/URIs; blocked spans surface auto-repair candidates.

## Immediate Tasks
1. **Front-door README** – position STM as the reality filter (hero paragraph + demo link). ✅
2. **Docs index** – maintain `docs/INDEX.md` with Guardrail Demo / Hallucination Control / Trading Heritage / API clusters. ✅
3. **Archive outputs** – keep generated artefacts in `results/` (stream JSONL, metrics, scatter); reference via docs index. 🔁 ongoing as we add new packs.
4. **Terminology hygiene** – patternability = structural rhythm, coherence = STM metric, hazard = lambda; reflected across notes (`semantic_coherence_hallucination_framework.md`, storyboard). ✅ initial pass.
5. **Automation hygiene** – `make clean` removes regenerated demo artefacts without touching curated data. ✅
6. **Truth-pack pipeline** – scripting to ingest arbitrary corpora → manifold + semantic projections + ANN artefacts. ✅
7. **Reality filter wrapper** – streaming span interceptor enforcing repeat ≥ r_min, hazard ≤ λ_max, cosine ≥ σ_min; performs twin-driven repairs. ☐
8. **Dashboard polish** – connect to new wrapper output (no hardcoded stream), expose twin links, KPI dials. ✅ (live metrics + twin citations)
9. **Calibration tooling** – lightweight percentile sweep + permutation validation for new packs. ☐
10. **Whitepaper note** – 6–8 page methods write-up once demo loop runs end-to-end. ☐

## Two-Week Delivery Plan (You are the technical hire)
- **Week 1 – Pack & Gate**
  - Curate truth-pack; run ingestion; expose `/seen`-style API returning repetitions, hazard, twins.
  - Establish baseline thresholds (r_min, λ_max, σ_min) via percentile sweeps; log stats.
- **Week 2 – Wrapper & Demo**
  - Hook LLM wrapper to gate (span-level admit/repair); wire dashboard to live stream (three panels + KPIs).
  - Re-run 3+1 question scenario; capture telemetry and citations; draft methods note.

## Risks & Mitigations
- **Last-mile scope**: clarify that filter validates spans, not entire arguments; demo includes trick question refusal.
- **Significance variance**: run permutation sweeps per pack; disclose weaker regimes.
- **Latency**: measure encode + ANN lookup; show ms/line in dashboard (already instrumented).

## Call to Action
- **Action**: Build the truth-pack ingestion + reality filter wrapper now.
- **Outcome**: deliver interactive demo + reproducible artifacts ready for investor diligence and customer pilots.
