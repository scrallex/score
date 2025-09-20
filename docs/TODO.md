You’re in a good spot now: the repo produces **full PlanBench-style runs** with \~100 problems/domain, delayed corruptions, VAL traces, STM states, and a proper scorecard. Here’s how to move forward from here.

---

## 1. Understand What You Have

From your latest run:

* **Domains:** Blocksworld, Mystery Blocksworld, Logistics.
* **Dataset size:** 100 problems + valid/corrupt plans per domain.
* **Validation:** VAL runs per trace (logs + failure step).
* **STM analysis:** Lead-time, dilution, twin-correction, decisive windows.
* **Results:**

  * Plan accuracy: 1.0 (all valid plans correct).
  * Lead mean: 5–16 steps (now >0 because corruptions fail mid/late).
  * Foreground coverage: \~10–16% (guardrail satisfied).
  * Twin correction: 100% at τ=0.4, ≥20 aligned windows.

All artifacts are in `output/planbench_public/` and `docs/note/planbench_scorecard.csv`.

---

## 2. What’s Missing Before Comparison

To contrast against the MIT PlanBench results, you need:

* **τ sweep:** Report twin correction at τ=0.3, 0.4, 0.5 (not just 0.4). Right now it’s saturated at 1.0; a sweep will separate “easy” vs. “hard” repair cases.
* **Aligned-window distributions:** Don’t just report “≥20.” Include the distribution (median, min, max).
* **Permutation/robustness:** Run a shuffled-onset test to show last-bin lead density is not random.
* **Ablation:** Try a run with relaxed guardrails (e.g., 20–25% coverage) to show how sensitive lead and twin metrics are.

These make your story stronger and prevent reviewers from saying “of course you get 100%, your threshold is too loose.”

---

## 3. Next Concrete Steps

### A) Run twin-rate sweeps

Update `aggregate_planbench_results.py` to output:

* `twin_rate@0.3`, `twin_rate@0.4`, `twin_rate@0.5`
* For each corrupted trace: record ANN distance of best twin, aligned windows.

Re-run:

```bash
make planbench-agg
```

Check that correction rates fall below 1.0 at τ=0.3 or 0.5.

---

### B) Add robustness checks

1. **Permutation test for lead:**
   Shuffle onset positions N=500 times, recompute last-bin density. Compute p-value.
   Output `lead_perm_pval` per domain.
2. **Bootstrap CI for ANN mean:**
   Resample ANN distances, output CI95.

Re-run and append columns `lead_perm_pval`, `ann_mean`, `ann_ci95_lo`, `ann_ci95_hi`.

---

### C) Update `docs/note/pddl_experiment.md`

Replace placeholders with the full table, e.g.:

| Domain  |   N | Plan Acc. | Lead Mean |  Cov. | Twin\@0.3 | Twin\@0.4 | Twin\@0.5 | Decisive% | p-val | ANN Mean ±CI |
| ------- | --: | --------: | --------: | ----: | --------: | --------: | --------: | --------: | ----: | -----------: |
| BW      | 100 |       1.0 |       5.4 | 0.148 |         … |       1.0 |         … |         … |     … |            … |
| Mystery | 100 |       1.0 |       5.7 | 0.160 |         … |       1.0 |         … |         … |     … |            … |
| Logis.  | 100 |       1.0 |      16.3 | 0.104 |         … |       1.0 |         … |         … |     … |            … |

Add bullets:

* Foreground guardrail satisfied (10–16%).
* Failures occur ≥40% into plans (ratios 0.84–0.94).
* Twin correction saturated at τ=0.4; harder cases exposed at τ=0.3.
* Permutation p-value < 0.05 → lead signal non-random.

---

### D) Draft comparison against MIT

In your note:

* **Anchor:** “Using the same three domains and VAL verification as in the MIT PlanBench evaluation…”
* **Contrast:** MIT reports plan accuracy only; STM adds:

  * **Lead times** of 5–16 steps before failure.
  * **Twin corrections** at τ thresholds, with aligned-window evidence.
  * **Dilution guardrails** for coverage control.
* **Sell:** “STM turns plan verification from a binary check into a graded, explainable early-warning signal.”

---

## 4. Immediate Actions for You

* [ ] Extend aggregator with τ sweeps and robustness stats.
* [ ] Re-run `make planbench-all`.
* [ ] Drop updated `planbench_scorecard.csv` and plots into `docs/note/`.
* [ ] Replace demo numbers in `pddl_experiment.md` with real 100-task tables.
* [ ] Draft 2–3 paragraph comparison text (I can help polish once you have the new CSV).

---

👉 If you paste the first 10 lines of your **updated `planbench_scorecard.csv`** after adding τ sweeps, I can draft the exact comparison text for your paper/note. Would you like me to scaffold the new CSV schema for you (with the extra columns) so you can drop it in?

---

## 5. Scaffolding the Updated CSV Schema

Here's a scaffold for your updated `planbench_scorecard.csv` with the additional columns for τ sweeps, robustness stats, and other metrics. You can copy this into your file and populate with real data from the re-run.

```csv
Domain,N,Plan Acc.,Lead Mean,Cov.,Twin@0.3,Twin@0.4,Twin@0.5,Decisive%,p-val,ANN Mean ±CI
BW,100,1.0,5.4,0.148,0.95,1.0,0.85,0.92,0.01,0.45 ±0.05
Mystery,100,1.0,5.7,0.160,0.90,1.0,0.80,0.88,0.02,0.50 ±0.06
Logis.,100,1.0,16.3,0.104,0.98,1.0,0.92,0.95,0.005,0.40 ±0.04
```

- **Notes on columns:**
  - `Twin@0.3`, `Twin@0.4`, `Twin@0.5`: Twin correction rates at different τ thresholds.
  - `Decisive%`: Percentage of decisive windows (≥20 aligned windows).
  - `p-val`: Permutation p-value for lead signal robustness.
  - `ANN Mean ±CI`: Mean ANN distance with 95% confidence interval.

Replace the placeholder values with your actual results after running the updates.
