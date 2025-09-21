# Guardrail Calibration (5% Target)

Commands executed on Ubuntu 24.04 droplet (virtualenv `.venv`):

```bash
.venv/bin/python scripts/calibrate_router.py output/planbench_public/gold_state.json \
  --target-low 0.05 --target-high 0.07 --output analysis/router_config_gold_5pct.json

.venv/bin/python scripts/calibrate_router.py output/planbench_public/invalid_state.json \
  --target-low 0.05 --target-high 0.07 --output analysis/router_config_invalid_5pct.json
```

Outputs: `analysis/router_config_gold_5pct.json` / `.coverage.json` and `analysis/router_config_invalid_5pct.json` / `.coverage.json` for the aggregated corpus, plus domain-specific calibrations:

```bash
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

Summary:

- **PlanBench (aggregate gold traces)** – thresholds: coherence ≥ 1.76e-4, entropy ≤ 0.999884, stability ≥ 0.4729 → coverage **11.54%**.
- **PlanBench (aggregate invalid traces)** – thresholds: coherence ≥ 1.99e-4, entropy ≤ 0.999845, stability ≥ 0.4729 → coverage **11.50%**.
- **Blocksworld (gold)** – coherence ≥ 5.57e-5, entropy ≤ 0.999725, stability ≥ 0.4660 → coverage **5.02%**.
- **Mystery Blocksworld (gold)** – coherence ≥ 6.19e-4, entropy ≤ 0.999535, stability ≥ 0.4577 → coverage **5.03%**.
- **Logistics (gold)** – coherence ≥ 8.32e-5, entropy ≤ 0.999824, stability ≥ 0.4802 → coverage **5.07%**.
- Invalid-trace calibrations show similar 5.0–5.1% coverage per domain (see corresponding JSON files).

The expanded quantile grid now reaches the 5% coverage target per domain, while the previously published
aggregate (gold/invalid combined) still sits around 11.5%. The domain-level configs and results feed the
updated guardrail sensitivity appendix and whitepaper discussion of remaining limitations.
