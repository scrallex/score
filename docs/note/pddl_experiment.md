# PDDL Structural-Verifier Experiment

## Purpose
- Compare STM dilution-driven verification against VAL feedback on PlanBench traces
- Quantify temporal lead and false/positive behaviour when structural anomalies precede logical violations
- Produce reusable registers (`gold_manifold.json`, `invalid_manifold.json`) for training loops and demos

## Data Pipeline
- Convert PlanBench VAL traces to structural token streams with `stm_adapters.pddl_trace.PDDLTraceAdapter`
- Script `scripts/planbench_to_stm.py` generates:
  - `gold/tokens/*.txt` and `invalid/tokens/*.txt`
  - STM analysis states (signals, scores, dilution summary)
  - Semantic token archives for future mutual-information studies
- Default window parameters: `window_bytes=512`, `stride=256`; adjust per domain complexity

## Dilution Metrics
- `path_dilution`: entropy of next-signature distribution; high values highlight ambiguous structural futures
- `signal_dilution`: token diversity inside a window; spikes indicate foreground churn
- `semantic_dilution`: `1 -` normalised mutual information between structural signatures and semantic labels
- All metrics exposed via `sep_text_manifold.dilution` and surfaced in analysis state summaries and the new `stm dilution` CLI command

## Verification Register Build
1. Run `scripts/planbench_to_stm.py --valid <path> --invalid <path> --output docs/planbench_stm`
2. Inspect generated states:
   - `docs/planbench_stm/gold_state.json`
   - `docs/planbench_stm/invalid_state.json`
3. Registers live in `*_manifold.json` and include per-window dilution payloads for streaming use

## Feedback Loop Integration (Next Steps)
- Feed dilution payloads into training harness to emit foreground/deferred signals alongside VAL verdicts
- Implement twin lookup on high-dilution windows to suggest corrective exemplars before VAL failure
- Log lead time between rising path/signal dilution and eventual VAL rejection to demonstrate structural advantage

## Testing
- Unit coverage added in `tests/test_dilution.py`
- CLI regression now exercises `stm dilution` to ensure state files contain dilution summaries when `--store-signals` is used
- Minimal three-domain PlanBench-style run captured in `docs/note/planbench_scorecard.csv`

## Current Results Snapshot

- **Domains covered:** Blocksworld, Mystery Blocksworld, Logistics (1 valid + 1 corrupted trace each)
- **Plan accuracy:** 100% per domain on the valid traces (baseline sanity check)
- **Lead-time coverage:** STM flagged every corrupted trace before or at the failure step (coverage = 1.0). Lead mean currently `0` bins because the toy problems induce immediate violations; expand the dataset to observe earlier alerts.
- **Twin correction:** All corrupted traces yielded at least one structural twin within distance ≤0.4 (correction rate = 1.0 in this seed set).
- **Artifacts:**
  - `output/planbench_public/gold_state.json` / `invalid_state.json`
  - `output/planbench_public/invalid/metrics/summary.json`
  - `docs/note/planbench_scorecard.csv`

```
# 1. Generate plan traces directly from VAL
PYTHONPATH=src scripts/val_to_trace.py domain.pddl problem.pddl plan.txt \
  --output traces/plan.trace.json

# 2. Inject a delayed-plan error (40–80% window)
scripts/inject_plan_corruption.py plan.txt --output plan.corrupt.txt --seed 7

# 3. Run STM lead/twin analysis across all traces
PYTHONPATH=src .venv/bin/python scripts/planbench_to_stm.py \
  --valid traces/plan.trace.json \
  --invalid traces/plan.corrupt.trace.json \
  --output output/planbench_public

# 4. Aggregate per-domain metrics for reporting
scripts/aggregate_planbench_results.py \
  --input-root output/planbench_public \
  --output docs/note/planbench_scorecard.csv
```

> These figures remain preliminary and drawn from handcrafted mini traces to validate the tooling. Scale to the full PlanBench splits (100 problems/domain) for publication-ready comparisons.

## Deliverables Checklist
- [x] Dilution metrics module with CLI inspection and streaming router integration
- [x] PDDL trace adapter + PlanBench exporter
- [ ] Training loop hook blending STM feedback with VAL (pending)
- [ ] Comparative ablation report (pending)
