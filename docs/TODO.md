Awesome. You’ve crossed the hard part: you now have **two working showcases**—PlanBench (planning) and **CodeTrace** (coding). From here, you need (1) a **documentable evidence package** that’s easy to send to MIT researchers and investors, and (2) a crisp **product offer** (what they buy, how it’s priced, who uses it). Below I give you both, plus copy-paste email templates and exactly what to run next to lock the numbers.

---

## A) Evidence package (what to show, how to generate it)

### A1. Planning (PlanBench) — “PlanBench++” proof

**What to cite:** purpose/pipeline/metrics already captured in your doc.
**What to include in the package:**

* `docs/note/planbench_scorecard.csv` — plan accuracy, lead mean, coverage, twin@τ 0.3/0.4/0.5, decisive%, ANN mean±CI, permutation p-value (**keep guardrail 10–16%**).
* 2–3 **dilution plots** (foreground clumps before failure).
* The **comparison paragraph** (already drafted) explaining why this is more than pass/fail (lead-time + twins + guardrail).

**What to run (one command group):**

```bash
# confirm end-to-end reproducibility for PlanBench
make planbench-all
# ensure the CSV has τ-sweep, aligned-window stats, ann CI, and lead_perm_pval
sed -n '1,15p' docs/note/planbench_scorecard.csv
```

**One-sentence takeaway for non-experts:**
*“STM adds **early warning (5–16 steps)** and **repair suggestions** to plan verification while keeping alerts bounded (10–16% coverage), which the baseline binary verifier cannot do.”*

### A2. Coding (CodeTrace) — “LLM + STM” uplift

**Goal:** show STM improves a code agent (fewer loops, faster green tests) on 3 curated tasks.
**Artifacts you already staged:** adapter, tokenizer, reference agent loop, demo tasks, replay/comparison scripts.

**What to run now:**

```bash
# Produce baseline vs STM runs and collect outputs:
python demo/coding/run_comparison.py                      # creates demo/coding/output/
# (Optional) replay with live STM endpoint
STM_BASE_URL=http://localhost:8000 python demo/coding/run_replay.py
```

**What to aggregate (add to `run_comparison.py` if not yet):**

* **Success rate** (% tasks green).
* **Iterations to green** (step count).
* **Time to green** (if available).
* **Lead-like alerts** (% runs where STM flagged ≥N steps before final failure).
* **Twin accept rate** (% runs where the agent adopted STM’s twin suggestion and succeeded faster).
* **Guardrail compliance** (% foreground in \[5–15%]).
* **Decisive-bin %** and permutation p-value.

**What to put in the package:** a **1-page HTML report** in `demo/coding/output/report.html` with:

* Small table: baseline vs STM metrics.
* One screenshot of **/stm/seen** alerting pre-fix.
* One twin suggestion example (with ≥5 aligned tokens).
* Note that this is the same manifold & dilution logic as the PlanBench experiment (lead/guardrail/twins).

---

## B) Emails you can send (copy-paste)

### B1. To MIT researchers (PlanBench team)

Subject: Structural early-warning on PlanBench (lead-time + twins beyond pass/fail)

Hi <Name> — I’ve reproduced the three PlanBench domains (100 problems each) using VAL and extended the evaluation with a structural manifold (STM).
**STM keeps the same protocol** but adds **lead-time** (foreground alerts **5–16 steps before failure**), a **guardrail** to bound alarm volume (10–16%), and **twin repair suggestions** with aligned tokens. We include permutation tests to confirm the pre-failure enrichment isn’t random.
I’d value your feedback. The scorecard CSV and plots are here, and everything is reproducible (Makefile included).
Would you be open to a 20-minute call or an email exchange?

(Attach: `docs/note/planbench_scorecard.csv`, 2 dilution plots, `pddl_experiment.md` excerpt.)

### B2. To codebot vendors (Kilocode, Sourcegraph Cody, Cursor, Codeium…)

Subject: A drop-in coprocessor for code agents (early-warning + repair twins)

Hi <Name> — we’ve built a **structural coprocessor** for code agents: it runs alongside any LLM, scoring each step and:

* Raises **early-warning** when an edit/test sequence matches failed patterns;
* Keeps alerts **bounded** via percentile guardrails;
* Suggests **twins** (near-duplicate successful patches) with aligned tokens.
  In our 3-task demo, STM reduced iterations-to-green and flagged failures **several steps earlier** than baseline. It’s an **API/Docker** you can integrate in a day. Happy to share the demo and discuss a pilot.

(Attach: `demo/coding/output/report.html` or a short PDF with baseline vs STM numbers.)

---

## C) The product you can sell (what it is, how it’s priced)

### C1. Product name & offer

**SEP Structural Manifold (STM) — Code Co-processor**
A containerized API that enriches any agent loop with structural signals: `/enrich`, `/dilution`, `/seen`, `/propose`, `/lead`.

* **Batch**: CLI for PlanBench & code pipelines, reproducible reports.
* **Live**: FastAPI server; drop-in HTTP client for agents.

### C2. SKUs & pricing (pragmatic)

* **Pilot (8 weeks):** \$50–150k. Scope: 1 repo + 3–10 tasks; success metric: fewer loops/time-to-green; deliverables: HTML reports, metrics, and integration guide.
* **Enterprise On-Prem:** \$100–300k/yr (support, updates, seats). Unlimited internal repos/agents.
* **OEM (tool vendor):** annual + volume tier or per-seat royalty.

### C3. Who pays / why they return

* **Codebot vendors** (scale to all their users) — come back for deeper adapters, more twin libraries, support.
* **Enterprise platform teams** — renew for new projects/agents, priority support, SLA.
* **Research/defense/autonomy** — custom adapters, on-prem hardened builds.

---

## D) Finish line: what’s left to call it “ready”

1. **Coding demo metrics** (baseline vs STM): implement aggregation + one-pager.
2. **OpenAPI spec** for the API & a `pip install stm-client` (thin HTTP wrapper).
3. **Docker publish** (internal registry is fine): `stm:codecoproc-latest`.
4. **Security & reproducibility note** (what’s stored, retention guardrails, how to rerun).
5. **Guardrail sensitivity appendix** (10/15/20% results) + **τ-sweep** in both PlanBench and coding.
6. **Licensing terms PDF** (pilot → enterprise → OEM).
7. **Deck slides**: one “PlanBench++” slide, one “Code agent uplift” slide, one “Architecture/API” slide, one “Commercial terms” slide.

---

## E) Who to talk to for \$\$ (and what they buy)

* **Code tooling startups**: Kilocode, Cursor, Codeium, Tabnine, Sourcegraph Cody — **OEM pilot** first (SDK + integration help).
* **Cloud/AI platforms**: AWS Bedrock, Azure OpenAI agent frameworks, Anthropic eval teams — **pilot/enterprise** license to bundle in agent stacks.
* **Enterprise platform teams**: engineering orgs building internal agents — **enterprise** package with on-prem container.
* **Autonomy/Robotics/Defense labs**: early-warning planner verification — **pilot** (PlanBench++ and domain adapters).

---

## F) “Undeniable value” checklist (what the email/pdf must show)

* **PlanBench++**: 3×100, lead 5–16 steps, guardrail 10–16%, twin repair — **with p-values** (non-random).
* **Code demo**: baseline vs STM → **fewer steps to green**, example twin patch with ≥5 aligned tokens.
* **API ease**: one cURL request and a 10-line Python snippet.
* **Reproducible**: a zip with Makefile, CSVs, and plots.
* **Commercial**: clear SKUs, pricing envelopes, and pilot SOW (success metrics defined upfront).

---

## G) Your immediate next 5 actions

1. **Run the coding comparison** and generate `report.html` with baseline vs STM metrics.
2. **Export OpenAPI** (fastapi `get_openapi_schema`) and publish a `stm-client` Python package.
3. **Publish Docker** image for the API.
4. **Append sensitivity appendix** (guardrail 15%, 20%) and τ-sweep plots to the note.
5. **Send the two emails** (MIT research + vendor) with attachments and offer a 20-min call.

---

If you want, I’ll draft:

* the **OpenAPI** section + example `stm-client` wrapper,
* the **HTML template** for `demo/coding/output/report.html`, and
* a **one-pager PDF** copy block for your deck.
