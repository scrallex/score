Perfect—now that you’ve bootstrapped seeds, run `planbench_to_stm.py`, and produced a demo scorecard, here’s a **tight, no-mystery checklist** to get from your toy run → a legitimate PlanBench-style experiment you can compare against the MIT evaluation.

---

## A) Data you still need (and how to structure it)

You already created toy PDDL & traces. Now build a **100-task/domain** pack mirroring the MIT setup:

```
data/planbench_public/
  blocksworld/
    domain.pddl
    problems/            # ~100 files
      p_0001.pddl ... p_0100.pddl
    plans_valid/         # planner outputs (one per problem)
      p_0001.plan ... p_0100.plan
    plans_corrupt/
      p_0001_corrupt.plan ... p_0100_corrupt.plan    # delayed failures (mid/late)
    logs_val/
      p_0001.plan.val.txt ... (VAL stdout/stderr)
    traces/
      p_0001.trace.json ...                          # per-spec JSON, see §C.3
  mystery_bw/  ... (same shape)
  logistics/   ... (same shape)
```

**Where to get problems**

* Blocksworld, Mystery Blocksworld, Logistics PDDL problem sets (publicly available in many PDDL suites; pick any canonical set with \~100 tasks each).
* If you don’t have ready packs, **upscale your generator** to emit 100 per domain with varied difficulty (object counts, goals, etc.). Keep the domains aligned with the names above so your scripts don’t need changes.

---

## B) Valid & corrupted plans (so you get non-zero lead)

* **Valid plans**: use any planner (Fast Downward, Pyperplan, LPG, etc.) to produce `.plan` files for each problem.
* **Corrupted plans**: start from the valid plan and inject errors **after a prefix of k steps** (so failure isn’t immediate). Examples:

  * **Delete** an action at position *t* (where *t* is 40–80% into the plan).
  * **Swap** two adjacent actions at *t*.
  * **Insert** an extra action that violates a precondition later.
* Keep **at least half** of corruptions failing **after mid-plan**; you’ll otherwise get lead=0.

> Rule of thumb: aim for 30–50% of corrupted traces failing in the last third of the sequence so your lead-time metric can show lift.

---

## C) VAL integration (objective step-wise labels)

1. **Install VAL** on your box (Ubuntu):

   ```bash
   sudo apt-get update
   sudo apt-get install -y cmake g++ flex bison     # build prereqs
   # If a package exists for your distro: sudo apt-get install -y val
   # Otherwise build from source (VAL’s README); produce a `val` binary in PATH.
   ```
2. **Validate** every plan:

   ```bash
   val -v domain.pddl problems/p_0001.pddl plans_valid/p_0001.plan \
     > logs_val/p_0001.plan.val.txt 2>&1
   ```
3. **Per-trace JSON** (your `planbench_to_stm.py` likes per-trace inputs). Store at `traces/p_0001.trace.json`:

   ```json
   {
     "domain": "blocksworld",
     "problem_file": "problems/p_0001.pddl",
     "plan_file": "plans_valid/p_0001.plan",
     "valid": true,
     "val_log": "logs_val/p_0001.plan.val.txt",
     "actions": ["(pick-up b)", "(stack b c)", "..."]
   }
   ```

   Do the same for `plans_corrupt/*` with `"valid": false`.

> TIP: add a Python helper `scripts/val_to_trace.py` that takes (domain, problem, plan) → runs VAL → writes the `.trace.json` (populates `actions[]` from the `.plan`, sets `valid` from VAL outcome, and records any failed step position).

---

## D) Run your STM pipeline at scale

1. **Build STM states + lead/twin/dilution per plan**

   ```bash
   PYTHONPATH=src .venv/bin/python scripts/planbench_to_stm.py \
     --input-root data/planbench_public \
     --domains blocksworld,mystery_bw,logistics \
     --out-root output/planbench_public \
     --plots
   ```

2. **Aggregate to a scorecard**

   ```bash
   PYTHONPATH=src .venv/bin/python scripts/aggregate_planbench_results.py \
     --in-root output/planbench_public \
     --out docs/note/planbench_scorecard.csv
   ```

The aggregator should compute for each domain (and split valid/corrupt):

* **Plan accuracy** (valid problems solved).
* **Mean lead-time** (bins) for corrupted traces (non-zero if failure isn’t immediate).
* **Twin correction rate** (% of corrupted traces where a twin within distance ≤ τ would fix the trajectory; start τ=0.4).
* **Coverage** (foreground fraction) and optional **decisive-bin %** (PD<0.3 & SD<0.4).

---

## E) Guardrail & thresholds (avoid degenerate coverage)

* After your per-domain calibrate step, target **foreground coverage** in **5–20%** band. If coverage=1.0, your thresholds are too loose; tighten using higher coherence percentile or lower entropy percentile.
* For **twins**: fix a threshold τ (e.g., 0.4) and a minimum **aligned windows** count (e.g., ≥20) so that twin suggestions are meaningful. Log both τ and alignment lengths in your results.

---

## F) Documentation updates

* **docs/note/pddl\_experiment.md**:

  * Replace “demo” with **public set** results; show a table per domain:

    | Domain      |   N | Plan Acc. | Mean Lead (bins) | Twin Corr. | Cov. (%) | Decisive-bin (%) |
    | ----------- | --: | --------: | ---------------: | ---------: | -------: | ---------------: |
    | Blocksworld | 100 |         … |                … |          … |        … |                … |
    | Mystery BW  | 100 |         … |                … |          … |        … |                … |
    | Logistics   | 100 |         … |                … |          … |        … |                … |

  * Add short paragraphs: (1) why lead>0 now (delayed corruptions), (2) what τ you used for twins, (3) coverage kept in guardrail band.

* Keep **planbench\_scorecard.csv** in version control and cite the file path beside each table (so reviewers can click and verify).

---

## G) Makefile (one-liners)

Add targets so you/reviewers can reproduce:

```makefile
planbench-val:
\tpython scripts/val_to_trace.py --root data/planbench_public --domains blocksworld,mystery_bw,logistics --out traces

planbench-stm:
\tPYTHONPATH=src .venv/bin/python scripts/planbench_to_stm.py --input-root data/planbench_public --domains blocksworld,mystery_bw,logistics --out-root output/planbench_public --plots

planbench-agg:
\tPYTHONPATH=src .venv/bin/python scripts/aggregate_planbench_results.py --in-root output/planbench_public --out docs/note/planbench_scorecard.csv

planbench-all: planbench-val planbench-stm planbench-agg
```

---

## H) Quick QA before you write the comparison section

* **Lead ≈ 0**? Then your corrupted plans still fail too early; push the injection deeper in the sequence.
* **Coverage too high/low?** Re-run calibration to hit 5–20%.
* **Twin rate suspiciously 100%?** Increase alignment requirement or lower τ; you want non-trivial matches.
* **Plots**: eyeball a few dilution/lead plots per domain to confirm pre-failure clumps are visible.

---

## I) “Against MIT” comparison: the safe narrative template

> “Using the same three domains (Blocksworld, Mystery BW, Logistics) and VAL verification procedure described in the PlanBench evaluation, we processed 100 tasks per domain with STM. On corrupted trajectories, STM flagged **foreground clumps** ahead of failure with a mean lead of **L** bins while keeping foreground coverage in the **5–20%** guardrail. Across domains, STM recovered **T%** of corrupted sequences via structural twins (ANN threshold τ=0.4; aligned windows ≥20). Plan accuracy tracked MIT’s base rates, while STM additionally provides **pre-failure lead** and **twin-based corrections** the verify-register setup does not quantify.”

Swap in your actual numbers once you run the public sets.

---

## J) If you want me to generate helpers

Say the word and I’ll draft:

* `scripts/val_to_trace.py` (domain/problem/plan → VAL run → `.trace.json`).
* `scripts/aggregate_planbench_results.py` (walks `output/planbench_public` → CSV).
* A tiny **corrupt-plan** tool that injects delayed errors at positions 40–80% of the plan.

You’re **one good dataset run** away from a credible comparison. The structure is in place—now feed it real PDDL sets, run VAL, and publish the scorecard.
