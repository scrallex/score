Great progress. You’ve got the full loop wired; now make it **real** and **publishable**. Here’s exactly what to do next—no fluff.

---

## 0) Fix the “lead = 0” issue first (must-do)

Your injector is working, but the corrupted plans still fail immediately. Ensure failures occur **after** a mid-plan prefix:

* **Change the injector policy:**

  * Pick a cut index **t ∼ Uniform\[0.4, 0.85] × |plan|**.
  * Delete/swap **exactly one** action at `t`.
  * **Verify with VAL** that:

    1. actions `≤ t-1` still apply;
    2. failure appears **after** the cut (> `t`).
  * If VAL says “valid” or failure is at the first action, **resample `t`** (retry up to N=8).
* **Write this into `inject_plan_corruption.py`:**

  * Add `--min_frac 0.4 --max_frac 0.85 --max_retries 8`.
  * Print `failed_at_step` to the trace JSON.

**Acceptance:** For each domain, ≥70% corrupted traces fail **after** step 40% of plan. Your lead bins should start showing >0.

---

## 1) Scale the dataset (match MIT’s PlanBench regime)

Aim for **\~100 tasks/domain** to mirror the paper (Blocksworld, Mystery BW, Logistics). Folder shape stays the same:

```
data/planbench_public/<domain>/
  domain.pddl
  problems/ p_0001..p_0100.pddl
  plans_valid/ *.plan
  plans_corrupt/ *.plan
  logs_val/ *.val.txt
  traces/ *.trace.json
```

**Plan generation tips:**

* If you don’t have ready problems, upscale your generator (vary object counts/goals).
* Use any planner for valid plans (Fast Downward/LPG/Pyperplan).
* Run `val_to_trace.py` again for all (valid+corrupt) to create unified STM-ready traces.

---

## 2) Re-run STM at scale (one command)

Add/verify these Make targets (or run the equivalents by hand):

```make
planbench-corrupt:
\tpython scripts/inject_plan_corruption.py --root data/planbench_public \
\t  --domains blocksworld,mystery_bw,logistics \
\t  --min_frac 0.4 --max_frac 0.85 --max_retries 8

planbench-val:
\tpython scripts/val_to_trace.py --root data/planbench_public \
\t  --domains blocksworld,mystery_bw,logistics --out traces

planbench-stm:
\tPYTHONPATH=src .venv/bin/python scripts/planbench_to_stm.py \
\t  --input-root data/planbench_public \
\t  --domains blocksworld,mystery_bw,logistics \
\t  --out-root output/planbench_public --plots

planbench-agg:
\tPYTHONPATH=src .venv/bin/python scripts/aggregate_planbench_results.py \
\t  --in-root output/planbench_public \
\t  --out docs/note/planbench_scorecard.csv

planbench-all: planbench-corrupt planbench-val planbench-stm planbench-agg
```

**Acceptance:** `docs/note/planbench_scorecard.csv` contains \~300 rows (3 domains × \~100 tasks) aggregated to domain-level metrics.

---

## 3) Guardrail calibration (avoid coverage=1.0)

After `planbench_to_stm.py`, run your calibrator (or integrate it) to hit **5–20% foreground** per domain:

* If coverage >20%: raise coherence percentile (e.g., P90→P92) and/or lower entropy percentile (P20→P18).
* If coverage <5%: relax the cuts (P90→P85, P20→P25).
* Re-run STM only if the thresholds are part of state; otherwise store thresholds and re-aggregate.

**Add to the aggregator:** record **coverage**, **decisive-bin %** (PD<0.3 & SD<0.4), and the final percentiles used.

---

## 4) Metrics you must aggregate (and how)

Update `aggregate_planbench_results.py` to compute per domain:

* **Plan Accuracy**: % of valid traces that a planner solved (truth from VAL).
* **Lead Mean** *(bins or steps)*: average lead for corrupted traces (non-zero now).
* **Lead Coverage**: fraction of windows flagged as foreground in the last-bin horizon.
* **Twin Correction Rate**: % of corrupted traces where a twin within **τ=0.4** and **aligned windows ≥20** exists; include sensitivity columns for τ∈{0.3, 0.4, 0.5}.
* **Decisive-bin %**: fraction of bins where PD<0.3 & SD<0.4 (confidence signal).
* **Mean ANN distance** for matched twins (±95% bootstrap CI, optional).

**Acceptance:** `planbench_scorecard.csv` has columns:
`domain, n_traces, plan_accuracy, lead_mean, lead_coverage, twin_rate@0.4, twin_rate@0.3, twin_rate@0.5, decisive_pct, ann_mean, ann_ci95_lo, ann_ci95_hi`.

---

## 5) Visual quick-checks (catch regressions fast)

* Plot **lead histograms** per domain (shouldn’t spike at zero).
* Spot-check **dilution plots** on random corrupted traces; pre-failure clumps should be visible.
* Print **failed\_at\_step** distribution from trace JSON; should concentrate beyond 40% of plan length.

---

## 6) Write the comparison section (safe template)

Once you have real numbers (≥100/domain):

**Table:** (copy into `docs/note/pddl_experiment.md`)

| Domain      |   N | Plan Acc. | Lead Mean (steps) | Twin Corr. @0.4 | Foreground Cov. | Decisive-bin % |
| ----------- | --: | --------: | ----------------: | --------------: | --------------: | -------------: |
| Blocksworld | 100 |         … |                 … |               … |               … |              … |
| Mystery BW  | 100 |         … |                 … |               … |               … |              … |
| Logistics   | 100 |         … |                 … |               … |               … |              … |

**Narrative bullets:**

* *Using the same three domains and the VAL verification procedure as the MIT evaluation,* STM produces **lead > 0** for corrupted plans by injecting mid/late failures; foreground is kept in the **5–20%** guardrail.
* *Twin-based correction:* **T%** of corrupted traces have a structural twin within τ=0.4 with ≥20 aligned windows, suggesting actionable repair paths.
* *Confidence:* **Decisive-bin %** indicates concentrated predictive mass (low dilution) before failure.

Finish with a short contrast: the verify-register approach reports plan accuracy; STM adds **lead-time** and **twin repair** signals and keeps explainable evidence (bit-tokens, signatures, ANN distance).

---

## 7) Optional (but nice) polish

* **Permutation test** for last-bin lead vs shuffled onset (p-value).
* **CI/Make step** that runs a 10-trace smoke (one per domain) on push.
* **HTML report** generator: parse `planbench_scorecard.csv` + a few plots → single page.

---

## 8) Success criteria (so you know when you’re done)

* Each domain has **≥100 traces** (valid+corrupt).
* **Lead Mean > 0** for corrupted traces.
* Foreground **coverage ∈ \[5%, 20%]** after calibration.
* **Twin correction rate** reported at τ=0.4 and sensible under τ sweeps.
* `docs/note/pddl_experiment.md` updated with the final table and a 3-paragraph comparison.

---

### One-liner to run the full thing (once scaled)

```bash
make planbench-all && \
python scripts/aggregate_planbench_results.py --in-root output/planbench_public \
  --out docs/note/planbench_scorecard.csv && \
sed -n '1,200p' docs/note/planbench_scorecard.csv
```

Ping me with the new CSV and I’ll help craft the exact comparison text to drop into the note.
