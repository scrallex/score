You don’t need the whole PlanBench stack in your head—here’s exactly what to grab and how to run a **PlanBench-style** experiment with your **QBSA/QFH manifold** using the `score` repo.

---

## 1) What data do I actually need?

PlanBench is just a standardized way to evaluate planning methods. In the MIT paper they evaluate on **three domains**: **Blocksworld**, **Mystery Blocksworld**, and **Logistics**, and then verify plans with the **VAL** plan validator (they also reason over **state→action→state** chains) . Concretely you need:

1. **PDDL domain files** (one per domain).
2. **PDDL problem files** (many per domain; different object counts/difficulties).
3. **Plans** (action sequences) for each problem: both **valid** and some **corrupted/invalid** ones (to test early detection & correction).
4. **Trace/feedback per plan** (ideally VAL output: which action was applicable, where it failed, etc.).

   * In the paper they use VAL to verify every step, and they structure reasoning as ⟨s₀, a₁, s₁⟩, … chains; we can mirror that mechanically with your STM pipeline .

> If you only have (1) and (2), you can generate (3) using any classical planner (or a trivial hand-written plan for tiny problems), and generate (4) by running VAL on (domain, problem, plan) as the paper does .

---

## 2) Where do I find these?

* **Domains/problems (PDDL):** start with small, known instances of **Blocksworld**, **Mystery Blocksworld**, **Logistics** (the paper’s evaluation is exactly those three) .

  * Any public PDDL set for these domains works; the goal is parity with those domains, not the exact same seed set.
* **Plans:**

  * **Valid**: produce via any off-the-shelf planner (or include the ground-truth plans bundled with the problems if you have them).
  * **Invalid (corrupted)**: clone a valid plan and introduce 1–2 controlled defects (e.g., drop a required pick-up, swap action order) to create predictable failures; VAL will annotate where they fail .
* **Trace/feedback:** run **VAL** on each (domain, problem, plan) to log per-step applicability/failures—the paper’s pipeline relies on VAL for objective verification and error typing; we exploit the same signal in STM .

> TL;DR minimal starter pack per domain:
> `domain.pddl`, a folder of `problem_*.pddl`, a folder of `plan_*.txt` (valid) + `plan_*.corrupt.txt` (invalid), and one VAL log or JSON per plan.

---

## 3) What format should the “trace” be for `scripts/planbench_to_stm.py`?

Your new script already expects **per-trace inputs** and writes **STM states + lead-time + dilution + twin summaries**. If you don’t have a strict schema yet, use this **minimal JSON** per plan (easy to generate):

```json
{
  "domain": "blocksworld",
  "problem_file": "problems/p_0003.pddl",
  "plan_file": "plans/p_0003_valid.txt",
  "valid": true,
  "val_log": "logs/p_0003_valid.val.txt",
  "actions": ["(pick-up b)", "(stack b c)", "(pick-up a)", "(stack a b)"]
}
```

…and the same for invalid/corrupted plans with `"valid": false` and their VAL log. Group these per domain in folders, e.g.:

```
data/planbench/blocksworld/
  domain.pddl
  problems/*.pddl
  plans/*.txt
  plans_corrupt/*.txt
  logs/*.val.txt
  traces/*.json           # one JSON per plan (as above)
```

Do the same for `mystery_bw/` and `logistics/`.

---

## 4) How to run it with your repo

Once the folders above exist:

```bash
# 1) Build STM states + lead/dilution/twin summaries per plan
python scripts/planbench_to_stm.py \
  --input-root data/planbench \
  --domains blocksworld,mystery_bw,logistics \
  --out-root output \
  --plots  # if matplotlib available

# 2) (Optional) aggregate to a single results CSV/JSON
python scripts/aggregate_planbench_results.py \
  --in-root output \
  --out docs/note/planbench_scorecard.csv
```

What you should see under `output/` now (as your script’s commit message implies):

* **Per-trace STM states** (JSON)
* **Lead-time** metrics (JSON)
* **Dilution plots** (PNGs)
* **Twin-correction summaries** (JSON) aggregated in `output/{gold,invalid}/…`

---

## 5) What numbers do we compare to?

The MIT PDDL-INSTRUCT paper reports **plan validity (accuracy)** across the **three domains** using PlanBench, and explicitly evaluates with **VAL** (100 tasks per domain) . For an apples-to-apples narrative:

* **Primary metric to mirror:** **Plan Accuracy** (% of problems with a valid plan), per domain.
* **Your STM extras:**

  * **Lead-time** (minutes/steps before failure/goal that your foreground density signals “high-coherence” regime).
  * **Twin-correction rate** (fraction of corrupted cases where your twin suggestion would have repaired the sequence).
  * **Dilution indicators** (median PD/SD in decisive bins).

Create a table like:

| Domain      | #Problems | Plan Accuracy | Mean Lead-time (bins) | Twin Correction Rate | Decisive-bin % (PD<.3 & SD<.4) |
| ----------- | --------: | ------------: | --------------------: | -------------------: | -----------------------------: |
| Blocksworld |       100 |             … |                     … |                    … |                              … |
| Mystery BW  |       100 |             … |                     … |                    … |                              … |
| Logistics   |       100 |             … |                     … |                    … |                              … |

Then write the comparison paragraph: *“Under the same PlanBench domains and VAL verification setup described in the MIT study (Fig. 1; Secs. 5–7), STM attains plan accuracy X/Y/Z and produces lead-time of L bins with twin-correction rate T%…”* .

---

## 6) If you don’t have ready-made PlanBench problems

You can still **bootstrap a small working set** to prove the pipeline:

1. **Blocksworld mini (5–10 problems):** hand-craft 3–5 tiny `problem_*.pddl`.
2. **Valid plans:** write short 2–6 action sequences.
3. **Corrupted plans:** delete or swap one action to make an invalid sequence.
4. Run **VAL** on each (domain, problem, plan) to produce the logs.
5. Create **trace JSON** entries per the schema above.
6. Run the two commands in §4 to generate STM outputs.
7. Once that works end-to-end, scale to Mystery BW and Logistics.

This gets you **plots and lead/twin summaries today**, without waiting for full PlanBench mirrors.

---

## 7) Quick glossary (to de-confuse terms)

* **PlanBench**: a benchmark suite + methodology to test planning; the paper uses it to evaluate three domains and measure plan accuracy **with VAL** .
* **Trace** (in our STM context): everything for a single plan attempt—domain, problem, action list, **VAL feedback** (per step), and our computed STM metrics (coherence, stability, entropy, rupture, dilution, lead-time, twins).
* **VAL**: a standard plan validator; the MIT experiments use it for stepwise verification and error typing; STM reuses the same signal to label foreground bins and twin corrections .

---

## 8) Your next moves

* **A.** Make the three domain folders with the minimal pack (domain.pddl, problems, valid & corrupted plans, VAL logs, simple trace JSONs).
* **B.** Run `planbench_to_stm.py` + aggregation as shown.
* **C.** Drop the aggregated CSV into your note; add a short “against MIT” comparison paragraph (same domains, same verifier).
* **D.** If the numbers look promising, scale the dataset (100 tasks each) and re-run.

If you want, paste me one tiny Blocksworld `domain.pddl` + a single `problem.pddl` + one valid `plan.txt` and I’ll give you the exact **trace JSON** and `VAL` invocation string to match your script’s expectations so you can run the full loop immediately.
