# Permutation Study – 20k Iterations (2025-09-21)

Commands executed from repo root inside `.venv` after calibrating the 5% guardrails:

```bash
for dom in blocksworld mystery_bw logistics; do
  .venv/bin/python scripts/run_permutation_guardrail.py \
    output/planbench_by_domain/$dom \
    analysis/router_config_${dom}_invalid_5pct.json \
    --iterations 20000 \
    --output docs/tests/permutation_${dom}_5pct.json
done

.venv/bin/python scripts/run_permutation_guardrail.py \
  output/planbench_public \
  analysis/router_config_invalid_5pct.json \
  --iterations 20000 \
  --output docs/tests/permutation_planbench_invalid_5pct.json
```

Summary statistics (weighted coverage, lead mean, permutation $p$ mean/min):

- Blocksworld: 6.89%, 4.45 steps, $p_\text{mean}=0.866$ (95% CI [0.836, 0.896]), $p_\text{min}=0.615$, precision=1.0.
- Mystery Blocksworld: 8.53%, 1.76 steps, $p_\text{mean}=0.872$ (95% CI [0.822, 0.921]), $p_\text{min}=0.250$, precision=1.0.
- Logistics: 4.41%, 2.86 steps, $p_\text{mean}=0.687$ (95% CI [0.614, 0.759]), $p_\text{min}=0.070$, precision=1.0.
- Aggregate PlanBench invalid: 5.47%, 7.59 steps, $p_\text{mean}=0.885$ (95% CI [0.865, 0.906]), $p_\text{min}=0.105$, precision=1.0.

Artefacts updated: `docs/tests/permutation_*_5pct.json`, `analysis/permutation_summary_5pct.csv`, and regenerated PlanBench figures in `docs/whitepaper/figures/` reflecting the 5% calibration.
