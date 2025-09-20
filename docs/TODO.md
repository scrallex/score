I’ve read your updated `pddl_experiment.md`. The good news is: the repo is now in a “publish-ready” state — you’ve got the full PlanBench pipeline (100 problems/domain, mid/late corruptions, VAL traces, STM analysis, aggregation) plus all the evidence artifacts. What you need now is clarity on **how to present and interpret this**. Here’s how I’d define it and what to do next:

---

## 1. What You’ve Built

* **Dataset:** Three canonical PlanBench domains (Blocksworld, Mystery Blocksworld, Logistics), each with 100 problems.
* **Processing pipeline:**

  1. Generate problems + valid plans.
  2. Inject mid/late corruptions (40–85% of plan length, retries until failure).
  3. Run VAL to validate every plan and record the failure index.
  4. Convert traces to STM token streams.
  5. Run `planbench_to_stm.py` to compute dilution metrics, foreground coverage, and twin suggestions.
  6. Aggregate into domain-level scorecards.
* **Metrics now logged:** Plan accuracy, mean lead time, coverage (guardrail \~10–16%), twin correction @τ=0.3/0.4/0.5, decisive-bin %, ANN mean ±CI, aligned-token stats, permutation p-value for lead.

This goes beyond the MIT “verify register” baseline, which only reports **plan validity**.

---

## 2. How to Frame the Results

* **Plan accuracy:** Matches MIT baseline (1.0 for valid plans).
* **Lead-time:** You can now show that STM consistently raises foreground alerts **5–16 steps before VAL failure**. MIT cannot report this; it’s your differentiator.
* **Twin correction:** At τ=0.4, you’re saturated at 1.0, but with the τ sweep you can show how stricter thresholds (τ=0.3) reduce corrections → this splits easy vs. hard repair cases.
* **Dilution guardrail:** Foreground coverage inside 5–20% (target achieved).
* **Permutation tests:** p-values << 0.05 confirm that the lead enrichment is non-random. This strengthens the claim that STM is predictive, not noise.

---

## 3. What’s Missing / Next Refinements

* **Twin robustness:** Report correction rates at τ=0.3 and τ=0.5 (you already have the columns, highlight them in the note).
* **Aligned-window evidence:** Add medians/min/max into `pddl_experiment.md` to demonstrate non-trivial overlaps.
* **Decisive-bin %:** Right now it reads near 0. Investigate if thresholds are too loose, or note that foreground is spread evenly (it’s fine if honest).
* **Sensitivity sweeps:** Try a guardrail at 15% or 20% to see if lead-time or twin correction changes. A single appendix table is enough.

---

## 4. How to Write the MIT Comparison Paragraph

In `pddl_experiment.md`, after your table, add:

> Using the same three domains and VAL verification as the PlanBench evaluation, STM achieves the same 100% plan accuracy on valid traces, but additionally surfaces structural early-warning signals. On corrupted traces, STM produces mean lead times of 5–16 steps with foreground coverage held to 10–16% by percentile guardrails. All corrupted traces identify structural twins at τ=0.4 (≥20 aligned windows), with robustness sweeps at τ=0.3/0.5 splitting easy vs. harder repair cases. Permutation tests (p<0.05) confirm that foreground clumps preceding failure are non-random. Unlike the binary “verify register” reported in the MIT study, STM provides graded, explainable pre-failure alerts and actionable twin-based corrections.

---

## 5. Immediate To-Dos

1. **Highlight τ sweeps**: update the scorecard table in `pddl_experiment.md` to show τ=0.3/0.4/0.5 side by side.
2. **Add aligned-token summary**: one line per domain with med/min/max overlaps.
3. **Note permutation p-vals**: emphasize they show significance.
4. **Draft the MIT comparison paragraph** (see template above).
5. **Optional:** run one guardrail sensitivity (15%) to show stability.

---

👉 If you like, paste me the first 15 lines of the updated `docs/note/planbench_scorecard.csv` and I’ll write the exact comparison paragraph ready to drop into your note. Would you like me to do that?
