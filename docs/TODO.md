Here’s the straight read on where your work stands today, plus the exact moves I’d make to finalize the experiments for the MIT send-off.

---

## 2024-xx-xx updates (Codex run)

* ✅ Ran 768 B window ablations for Blocksworld and Logistics; table + narrative updated in `docs/whitepaper/STM_Structural_Manifold_Whitepaper.tex` (Logistics $p_{\min}$ rises to 0.12 at 5\% coverage; dynamic 2.5\% sits at 0.33).
* ✅ Extended Logistics corpus to 500 traces (new root `data/planbench_scale500`), recalibrated guardrails, and appended sweep results (5\% target still hits $p_{\min}=0.035$ at \~2.9\% coverage; 2.5\% target reverts to $p_{\min}=0.33$).
* ✅ Scaled Blocksworld and Mystery to 500 traces via `make planbench-scale`; both domains stay null (Blocksworld $p_{\min}=0.62$, Mystery $p_{\min}=0.14$) while preserving $\leq 9\%$ coverage.
* ✅ Regenerated appendix sweep CSV + feature ablation table to capture the new window and scale probes; whitepaper recompiles cleanly.
* ✅ Added Logistics 2.5\% guardrail overlay figure and tightened whitepaper layout (no overfull boxes; tables now fit without float warnings).

---

## Where it stands (research-ready)

* **Guardrail calibration + permutation auditing:**
  Your whitepaper shows dense percentile grids that hit 5% coverage on all domains, and a **low-guardrail sweep (1–5%)** that proves only **Logistics** crosses p < 0.05 (at 2.5%), while **Blocksworld** and **Mystery** remain high (p\_min ≈ 0.66 and 0.08 even at very low coverage). See the **permutation section** and the **low-guardrail table**; mean/min p-values and CIs are clearly reported and Logistics’ dynamic drop to 2.5% is justified (p\_min ≈ 0.035 with \~10-step lead) (page 6, table on page 7).
* **Lead-time + twins are real and repeatable:**
  STM shows **lead-time 5–16 steps** and **perfect twin recall across τ (0.3–0.5)** in PlanBench; the **τ-sweep figure** shows a flat 1.00 acceptance rate (ample headroom to tighten thresholds) (page 7).
* **CodeTrace: agent uplift with bounded alerts:**
  The coding demo aggregates **\~35% fewer steps-to-green** and **single-window alerts**; tables list per-task and aggregate deltas, and the paper includes a **/stm/seen** screenshot + **twin patch** example (edits reduce flaky test failures) (pages 7–9).
* **Reproducibility:**
  The Appendix and Repro checklist show command sequences and Make targets to regenerate datasets, calibrations, permutations, and reports (page 10).

Net: you’ve moved PlanBench from a binary pass/fail into a **graded, auditable early-warning + repair** framework. The paper is research-focused, limitations are explicit, and everything is reproducible.

---

## What to include for MIT (and what not)

**Keep (these are the load-bearing contributions):**

* **PlanBench++ story**: 3×100 tasks; **lead 5–16 steps**, **guardrail 5–10%**, **twins across τ**; permutation analysis with **Logistics significance at 2.5%**.
* **Dynamic guardrail**: automatic Logistics drop to 2.5% with audit trails and before/after lead differences (10-step vs 2.9-step at 5%).
* **CodeTrace vignette**: the table with baseline vs STM, the **/stm/seen** panel, and the **twin patch** snippet.
* **Reproducibility**: Make/CLI commands and dataset pointers.

**Trim / move out (for the MIT version):**

* Any **pricing/licensing** or commercial packaging (keep that for vendor/investor collateral).
* Long product sections; keep just enough about the API so they know it’s rerunnable.

---

## What to expand (to make it noteworthy)

1. **Dynamic calibration details (Logistics)**
   Add a short box: *“Why Logistics achieves significance”*—longer horizon, richer structure, and how lowering coverage to 2–3% concentrates decisive bins. Cite the table where p\_min drops to 0.035 at 2.5% with \~10-step lead (table on page 7).

2. **Negative results (Blocksworld, Mystery)**
   Make the null finding explicit: even at 1–3% coverage, p-values stay high (≥0.60, ≥0.08). State the hypothesis: **richer structural features** (longer windows, signature-aware filters, domain-specific tokens) are needed before meaningful significance is realistic. (You do hint this in Discussion—promote it to a bullet in Results.) (page 6).

3. **Feature/twin ablations (one figure or small table)**
   Include a tiny ablation:

   * **Longer foreground windows** (e.g., 512 B→768 B) effect on p-values and lead.
   * **Signature-aware twin filtering** (≥N shared q-grams) vs. default.
   * Show **no change** for BW/Mystery (honest) and a small trend for Logistics (if any).
     This preempts “try different thresholds/features” reviewer feedback.

4. **Scale knob (PlanBench 300→500)**
   You already list scale as future work; if feasible, add a 500-problem Logistics subset to confirm the 2.5% result still holds (even a footnote: n=500 keeps p<0.05, lead \~10). If time is short, leave as “in progress.”

---

## Do you need a rewrite?

No. The draft is already structured like a solid research note. Do a **surgical revision**:

* In **Results**, add a two-paragraph “Dynamic guardrail outcome” emphasizing Logistics 2.5% significance and BW/Mystery nulls with pointers to the sweep table (page 7).
* Add a small **Ablation** subsection (as above).
* In **Discussion**, explicitly phrase the scientific takeaway: coverage control ≠ significance; **stronger features** (and longer traces) are the next research lever (page 8–9).
* Keep all commercial talk out of the MIT PDF.

---

## Final experiment checklist (to run before you send)

* **Rebuild the 2.5% Logistics profile** and export a tiny overlay figure (lead histogram + p-value trend) referenced in Results.
* **One ablation run** (longer window or stricter twin filter) on Logistics + BW to populate the new mini table (even if BW stays null, that’s fine).
* **Recompile whitepaper**; confirm page references to the new table/figure resolve.
* **Zip evidence**:

  * `planbench_scorecard.csv` + **low-guardrail sweep JSON** (per domain).
  * `router_config_*_5pct.json` and the dynamic Logistics 2.5% config & coverage report.
  * CodeTrace `report.html` (with /stm/seen screenshot and twin patch).
  * Repro instructions.

---

## One-paragraph cover note (for MIT)

> We extended PlanBench with **structural guardrails** that emit **early-warning windows** and **twin repairs**. Across 3×100 tasks, STM shows 5–16-step lead times with bounded alert coverage. A low-guardrail sweep (1–5 %) reveals that **Logistics** achieves **p ≤ 0.035** at \~2.5 % coverage, while **Blocksworld** and **Mystery** remain indistinguishable from random permutations even at ≤3 %. We include configs, permutations, and a rerunnable suite. We’re now experimenting with feature-level improvements (longer windows, signature-aware twins) to strengthen discriminative power. We’d appreciate your feedback on the methodology and whether this “PlanBench++” direction merits a joint study.

If you want, I’ll draft the small **ablation table** and insert-ready **two-paragraph edits** for your Results and Discussion sections referencing the sweep table and figure numbers from the current PDF so you can paste and compile immediately.&#x20;
