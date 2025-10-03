# Every Line Must Carry Its Receipts — Operational Roadmap

## North Star & Definition of Done

- **North Star:** Deliver an interactive, data-agnostic reality filter demo plus a reproducible methods note.
- **Definition of Done (v1):**
  - `make semantic-guardrail-demo` runs end-to-end on any truth-pack (no hardcoding) and drives the three panels with admit/repair decisions, twins, and KPIs.
  - A 6–8 page methods note with figures (manifold build, threshold sweeps, permutation results, KPI deltas) and exact reproduction commands.
  - A two-week pilot playbook covering truth-pack prep → demo run → report export.

---

## Repo Hygiene & Documentation (1–2 days)

- **README.md (root):** concise positioning, one-command demo, link to docs index.
- **`docs/INDEX.md`:**
  - Getting started (venv, data, `make semantic-guardrail-demo`).
  - Running custom truth-packs.
  - Threshold tuning guidance.
  - Artefact locations.
  - Link to methods note.
- **Doc set:**
  - `docs/note/semantic_manifold_operational_roadmap.md` (live roadmap).
  - `docs/note/reality_filter_demo.md` (10–12 minute operator runbook).
  - `docs/whitepaper/reality_filter_methods.md` (methods note).
- **Project map:**
  - Guardrail section first; archive stale scripts or folders.

**Make targets to implement/verify**

| Target | Description |
| --- | --- |
| `make semantic-guardrail-demo` | Full demo loop (truth-pack → stream → dashboard). |
| `make clean-demo` | Purge generated truth-pack artefacts and results. |
| `make pack PACK=<path> seeds="..." ext=md` | Wrapper around `scripts/reality_filter_pack.py`. |
| `make stream PACK=analysis/truth_packs/<pack>` | Run `scripts/reality_filter_stream.py` on prepared pack. |
| `make report` | Collate KPIs/figures into `results/report/`. |

---

## Engineering Backlog (ordered)

1. **`/seen` Service Boundary (must) ✅**
   - Interface:
     - Request `{text, window_bytes, seed_vec?, pack_manifest}`.
     - Response `{repetitions, lambda, coherence, stability, entropy, signature, semantic_score, twins[], decisions{repeat_ok,hazard_ok,semantic_ok}}`.
   - Deliver a fast in-process engine (`TruthPackEngine`) plus a FastAPI shim.
   - Acceptance: ≥1k span calls/sec on sample pack; export latency histogram.

2. **LLM Wrapper (must) ✅ (span source swap in CLI)**
   - Span sources: `SimSpanSource` (existing JSON) and `LLMSpanSource` (OpenAI/HF/local) with question context.
   - Pipe spans through `TruthPackEngine.evaluate_span`; emit admit/repair/decline.
   - Acceptance: Source selected via CLI; dashboard reflects live gating in <200 ms/line on sample pack.

3. **Twin-Based Auto-Repair (should) ✅ (deterministic fallback + LLM prompt hook)**
   - Replace blocked spans with top-K twin snippets via constrained paraphrase prompt (cite tokens preserved) and re-check.
   - Emit before/after plus twin URIs.
   - Acceptance: `repair_yield` KPI calculated and charted.

4. **Semantic Bridge Controls (should)**
   - Dual seed families (factual vs novelty); compute semantic margin and apply `σ_min` to margin.
   - CLI flags to set seed lists; log margin.

5. **Incremental Pack Updates (nice)**
   - Append new docs to manifolds without rebuilding from scratch; maintain window IDs.
   - Acceptance: incremental runs <30 % of fresh ingest time on small delta packs.

---

## Dashboard Polish

- **Panels:**
  - Left: naïve semantic (tooltip shows span + top seeds).
  - Middle: naïve structural (coherence, stability, hazard badge).
  - Right: hybrid scatter with span details, decisions, twins, citations, latency.
- **KPI bar:** Hallucination rate, repair yield, citation coverage, latency (ms/line).
- **Controls:** Sliders for `r_min`, `λ_max`, `σ_min` with replay buffer recompute.
- **Logging:** `results/semantic_guardrail_metrics.json` updated on every run.

---

## Experiments (reproducible)

### Truth-Packs
1. Internal `docs/note` pack (dogfood).
2. Public policy/manual pack (shareable).

### Threshold Sweeps (per pack)
- Grid: `r_min ∈ {1,2,3}`, `λ_max ∈ {0.12,0.15,0.18,0.22,0.25}`, `σ_min ∈ {0.15,0.20,0.25,0.30}`.
- Output CSV → `results/sweeps/<pack>.csv`.

### Permutation Tests
- 20k shuffles to estimate p-values across 1–5 % coverage bands.

### Ablations
- Structure-only vs semantic-only vs hybrid.
- With vs without twin repair.
- Window bytes 256 vs 768.

### KPIs (per question set)
- `hallucination_rate = blocked / total`.
- `repair_yield = repaired / blocked`.
- `citation_coverage = approved_with_twins / approved`.
- `latency_ms_p50`, `latency_ms_p90`.
- Manual `precision@k` for citation spot-check.

**Make targets to add:** `make sweep`, `make permutation`, `make ablations`, `make report`.

---

## Telemetry Schemas

**Stream JSONL (`results/semantic_guardrail_stream.jsonl`):**
```json
{
  "ts": "...",
  "qid": "Q1",
  "question": "...",
  "span": "...",
  "decisions": {"repeat_ok": true, "hazard_ok": true, "semantic_ok": true, "admit": true},
  "metrics": {"repetitions": 3, "lambda": 0.16, "coherence": 0.72, "stability": 0.69, "semantic": 0.31, "margin": 0.22},
  "twins": [{"uri": "doc://...", "offset": 1234, "distance": 0.18}],
  "action": "emit | repair | decline",
  "latency_ms": 87
}
```

**Metrics rollup (`results/semantic_guardrail_metrics.json`):**
```json
{
  "pack": "analysis/truth_packs/...",
  "thresholds": {"r_min": 2, "lambda_max": 0.18, "sigma_min": 0.25},
  "kpis": {
    "blocked": 12,
    "approved": 88,
    "repaired": 8,
    "hallucination_rate": 0.12,
    "repair_yield": 0.67,
    "citation_coverage": 0.89,
    "latency_ms_p50": 74,
    "latency_ms_p90": 128
  }
}
```

---

## Methods Note Outline (6–8 pages)

1. Motivation: workslop & hallucinations → need receipt-backed responses.
2. Structural admission: echo + hazard rule (QFH/STM).
3. Target manifold & twins: ingestion pipeline, ANN/q-gram blend, acceptance curves.
4. Reality filter demo: pack scope, thresholds, span gate, auto-repair loop.
5. Results: KPIs for two packs, ablations, sweeps, permutation p-values.
6. Limits & risks: span-level validation, geometry sensitivity, latency trade-offs.
7. Appendix: commands (Make targets), seed lists, JSON schemas, reproducibility notes.

---

## CI/CD & Reproducibility

- **GitHub Actions:**
  - `lint+tests` (black, ruff, pytest).
  - `demo-smoke` (small pack, 3 spans) to protect the workflow.
  - `build-report` on tags; publish `results/report/*` artefacts.
- **Determinism:** pin model versions & RNG seeds; emit per-run manifest.
- **Issue templates:**
  - Experiment request (hypothesis, thresholds, pack, KPI).
  - Bug report (repro steps, pack path, JSON lines).
  - Documentation update (page, owner, status).

---

## Acceptance Checklist (Before Demoing)

- [ ] `make semantic-guardrail-demo` runs clean on two distinct packs.
- [ ] Three panels live; sliders recompute; twins clickable; KPIs update.
- [ ] `results/report/` contains scatter, KPI bar, sweeps CSV, permutation plots, latency histograms.
- [ ] Methods note compiles with figures and exact commands.
- [ ] Trick question is visibly declined or repaired (no hardcoding).

---

## Next Focus for Codex

1. `/seen` interface + FastAPI shim.
2. LLM span source integration.
3. Twin-based auto-repair with constrained paraphrase loop.
4. Threshold sweeps + permutation targets, generate first report across two packs.
5. Finalise docs (index, runbook, methods note).
6. Harden telemetry (latency budgets / alerts).
7. Stretch: twin-constrained decoding + coherence reward for deeper integration.
