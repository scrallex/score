# Guardrail Calibration (5% Target)

## Router calibration runs

Calibrations ran on the Ubuntu 24.04 droplet inside `.venv`, using the densified percentile grid baked into `scripts/calibrate_router.py` (coherence 55–99th, entropy 2–60th, stability optional/55–90th). The tighter sweep made the 5% coverage window reachable across every corpus.

```bash
.venv/bin/python scripts/calibrate_router.py output/planbench_public/gold_state.json \
  --target-low 0.05 --target-high 0.07 \
  --output analysis/router_config_gold_5pct.json

.venv/bin/python scripts/calibrate_router.py output/planbench_public/invalid_state.json \
  --target-low 0.05 --target-high 0.07 \
  --output analysis/router_config_invalid_5pct.json

for dom in blocksworld mystery_bw logistics; do
  .venv/bin/python scripts/calibrate_router.py \
    output/planbench_by_domain/$dom/gold_state.json \
    --target-low 0.05 --target-high 0.07 \
    --output analysis/router_config_${dom}_gold_5pct.json
  .venv/bin/python scripts/calibrate_router.py \
    output/planbench_by_domain/$dom/invalid_state.json \
    --target-low 0.05 --target-high 0.07 \
    --output analysis/router_config_${dom}_invalid_5pct.json
done
```

Each invocation writes both the router configuration (`*.json`) and the associated coverage/percentile dump (`*.coverage.json`). Verified coverage values:

- **Aggregate gold** (`analysis/router_config_gold_5pct.json`) → min_coh 8.32e-5, max_ent 0.99970, min_stab 0.47096, coverage **5.09%**.
- **Aggregate invalid** (`analysis/router_config_invalid_5pct.json`) → min_coh 1.16e-4, max_ent 0.99972, min_stab 0.47582, coverage **5.01%**.
- **Blocksworld gold** (`analysis/router_config_blocksworld_gold_5pct.json`) → coverage **5.02%**; invalid set at **5.10%**.
- **Mystery Blocksworld gold** (`analysis/router_config_mystery_bw_gold_5pct.json`) → coverage **5.03%**; invalid set at **5.07%**.
- **Logistics gold** (`analysis/router_config_logistics_gold_5pct.json`) → coverage **5.07%**; invalid set at **5.05%**.
- Percentile tables in the `.coverage.json` companions confirm smooth quantile ramps for coherence, entropy, and stability.

The 5% guardrail now holds per domain and on the aggregate corpora without sacrificing the ANN trigger defaults (`min_sig_qgrams = 2`, `max_ann_dist = 0.2`). Permutation p-values, however, remain elevated (0.85–0.99) and are still called out as a limitation in the whitepaper refresh.

## Regenerated STM states

Regenerated domain-specific STM artefacts to ensure router configs reference the latest tokens, states, and lead/twin metrics:

```bash
for dom in blocksworld mystery_bw logistics; do
  .venv/bin/python scripts/planbench_to_stm.py \
    --input-root data/planbench_public \
    --domains $dom \
    --output output/planbench_by_domain/$dom \
    --window-bytes 256 \
    --stride 128 \
    --path-threshold 0.10 \
    --signal-threshold 0.10 \
    --twin-distance 0.40 \
    --twin-top-k 3 \
    --verbose
done
```

Each run produces per-trace token archives (`tokens/`), STM states (`gold_state.json`, `invalid_state.json`), detailed lead/twin metrics under `invalid/metrics/`, and an execution manifest (e.g. `output/planbench_by_domain/blocksworld/run_summary.json`).

## Permutation tests

Automated guardrail permutation tests with the new 5% configs to quantify how often a random alert schedule would outrun the observed lead times:

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

Highlights (weighting coverage by total windows):

- Guardrail sweep automation: `scripts/guardrail_sweep.py` records coverage vs $p$ across 2–8% (see `docs/note/appendix_guardrail_sweep.csv`).
- 95% CI on permutation means: Blocksworld [0.836, 0.896], Mystery BW [0.822, 0.921], Logistics [0.614, 0.759]; aggregate PlanBench stays [0.865, 0.906].
- Alert precision (share of alerts fired before the final failure) is **100%** across domains, so added sensitivity will require features beyond timing.
- **Blocksworld invalid** (`docs/tests/permutation_blocksworld_5pct.json`) → coverage **6.9%**, mean lead **4.4** steps (alerts on 47/100 traces), mean permutation $p$ **0.87** (min 0.62).
- **Mystery Blocksworld invalid** (`docs/tests/permutation_mystery_bw_5pct.json`) → coverage **8.5%**, mean lead **1.8** steps (25/100 traces), mean $p$ **0.87** (min 0.25).
- **Logistics invalid** (`docs/tests/permutation_logistics_5pct.json`) → coverage **4.4%**, mean lead **2.9** steps (43/100 traces), mean $p$ **0.69** (min 0.070).
- **Aggregate PlanBench invalid** (`docs/tests/permutation_planbench_invalid_5pct.json`) → coverage **5.5%**, mean lead **7.6** steps (alerts on 158/300 traces), mean $p$ **0.89** (min 0.10).

Even with the calibrated 5% coverage, permutation significance remains weak—95% CIs on the permutation means stay above 0.61 and alert precision is 1.0 across domains—reinforcing the roadmap item to expand corpora and harden the guardrail heuristics.


## Whitepaper refresh

Updated `docs/whitepaper/STM_Structural_Manifold_Whitepaper.tex` to cite the successful 5% calibrations while noting that permutation significance remains weak. Recompiled the PDF so the guardrail appendix, figures, and discussion pick up the new numbers:

```bash
latexmk -pdf -silent docs/whitepaper/STM_Structural_Manifold_Whitepaper.tex
```

The appendix now points to the `analysis/router_config_*_5pct*.json` artefacts, and the discussion section highlights the next steps (per-domain permutation automation, broader PlanBench/CodeTrace datasets, adapter expansion).
