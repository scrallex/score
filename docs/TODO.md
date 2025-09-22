Below is a detailed, step‑by‑step plan that your Codex bot can follow on the droplet under `/sep` to use real data, refresh all artifacts, and continue assembling the QFH foundation paper.  These instructions assume that the new JSON parsing code has been compiled (as you noted with `cmake --build build --target manifold_generator`) and that the `.env` file in `/sep` contains valid OANDA and Valkey credentials.

---

### 1. Prepare the environment

1. **Move to the SPT workspace**

   ```bash
   cd /sep
   ```

   The `spt` repository is the root here; the `score` repo is a subdirectory.

2. **Load environment variables**
   Source the `.env` file so the Python scripts can access your OANDA API key and Valkey URL:

   ```bash
   set -a
   source .env
   set +a
   ```

3. **Rebuild the native tools** (if you haven’t already)

   ```bash
   cmake --build build --target manifold_generator
   ```

4. **Ensure the services are up**
   You already ran `deploy.sh`, but it’s good to verify:

   ```bash
   docker compose -f docker-compose.hotband.yml ps
   ```

   You should see `sep-backend`, `sep-valkey`, `sep-candle-fetcher`, and related services all “Up” and healthy.  The trading backend uses the improved parsing helpers to deserialize candles so numeric strings will no longer cause a `nlohmann::json` type error.

---

### 2. Prime Valkey with real market data

The prime script reads historical candles via your OANDA credentials, creates manifolds, stores them to Valkey, and writes JSON snapshots in `output/warmup/<PAIR>/<DATE>.json`.  Run it for each instrument over a sufficient window (e.g., 30 days).

Here is an example loop for eight FX pairs (EUR/USD, USD/JPY, GBP/USD, EUR/JPY, USD/CAD, NZD/USD, AUD/USD, USD/CHF):

```bash
cd /sep
for pair in EUR_USD USD_JPY GBP_USD EUR_JPY USD_CAD NZD_USD AUD_USD USD_CHF
do
    python scripts/ops/prime_qfh_history.py \
      --instrument $pair \
      --days 30 \
      --store-manifold-to-valkey \
      --output-dir output/warmup  # optional: explicit output location
done
```

* This command fetches 30 days of 1‑minute candles per pair, converts them to bitstreams, computes the QFH metrics via the native engine (using the robust numeric parsing functions), builds signatures and repetition counts, and writes JSON files.
* Watch for log messages; if you see `terminate called after throwing…`, revisit your `.env` credentials and ensure the new parsing code has been picked up (the fix should prevent those errors).

---

### 3. Export snapshot CSVs for the figures

After the prime job completes successfully for a pair, export a 30‑day snapshot from Valkey for plotting:

```bash
for pair in EUR_USD USD_JPY GBP_USD EUR_JPY USD_CAD NZD_USD AUD_USD USD_CHF
do
    python scripts/ops/export_manifold_snapshots.py \
      --instrument $pair \
      --minutes $((30*24*60)) \
      --out output/manifolds_native/${pair}_snapshots.csv
done
```

Each CSV will contain timestamped rows with `coherence`, `entropy`, `stability`, `rupture`, `lambda_hazard`, and `repetition.count_1h`.  These will be used for the FX plots.

---

### 4. Regenerate PlanBench/Logistics artifacts (done, but repeat if necessary)

You already reran all PlanBench exports and calibration scripts with `--use-native-quantum` enabled.  If you need to refresh them after the parsing fix:

```bash
# Example for logistics domain
python scripts/planbench_to_stm.py \
    --domain logistics \
    --use-native-quantum \
    --output output/planbench_by_domain/logistics

python scripts/enrich_features.py \
    output/planbench_by_domain/logistics/gold_state.json \
    --output output/planbench_by_domain/logistics/gold_state_logistics_native.json \
    --features causal logistics --blend-metrics --use-native-quantum

python scripts/experiments/build_causal_domain.py \
    --domain logistics \
    --include-logistics \
    --use-native-quantum \
    --output output/planbench_by_domain/logistics_enriched_native

python scripts/calibrate_router.py \
    --domain logistics \
    --input output/planbench_by_domain/logistics_enriched_native \
    --use-native-quantum \
    --output analysis/router_config_logistics_enriched_native.json
```

This pipeline ensures that logistics states contain the native metrics and that guardrails are calibrated on those metrics.

---

### 5. Refresh synthetic and bridge datasets

* Run the synthetic generator to produce canonical event histograms:

  ```bash
  python scripts/experiments/qfh_synthetic.py \
    --output results/qfh_synthetic_native.json
  ```

* Recompute bridge statistics with the new PlanBench and FX data:

  ```bash
  python score/scripts/compute_bridge_metrics.py \
    --planbench output/planbench_by_domain/logistics/gold_state_logistics_native.json \
    --fx output/manifolds_native/EUR_USD_snapshots.csv \
    --output-metrics docs/note/bridge_metrics.json \
    --output-contingency docs/note/bridge_contingency.json
  ```

Replace `compute_bridge_metrics.py` with the actual script name used to recompute bridge metrics.

---

### 6. Generate updated figures and tables

Once all datasets (synthetic, PlanBench native, FX CSVs, bridge metrics) exist, regenerate the figures for your foundation paper:

```bash
python score/scripts/plot_whitepaper_figures.py \
  --synthetic results/qfh_synthetic_native.json \
  --planbench output/planbench_by_domain/logistics/gold_state_logistics_native.json \
  --fx output/manifolds_native/EUR_USD_snapshots.csv \
  --bridge docs/note/bridge_metrics.json
```

* If the script expects a warmup directory, ensure it points at `output/warmup` (where the JSONs from step 2 live).
* This will produce new histograms of QFH events, metric distributions and the updated bridge scatter/correlation.

Regenerate LaTeX tables (if applicable) using any table-generation scripts you have (e.g. `generate_receipt_tables.py` or equivalents).

---

### 7. Compile the foundation paper

1. Edit `docs/whitepaper/QFH_Manifold_Foundation.tex` to insert the newly generated figures (`fig1_*`, `fig2_*`, etc.) and update sections on experiments and results.
2. Include narrative explaining the robust JSON parsing fix: the new helper functions accept numeric strings, ISO‑8601 timestamps and nested mid/bid/ask payloads and avoid the previous type errors.
3. Run LaTeX to produce the PDF:

   ```bash
   cd docs/whitepaper
   latexmk -pdf QFH_Manifold_Foundation.tex
   ```
4. Review the PDF, adjust figure sizes if necessary, then commit the updated `.tex` and `pdf`.

---

### 8. CI and testing

1. Modify your CI configuration (e.g. GitHub Actions or your `Makefile`) to run:

   ```bash
   pip install -e ./score[native]
   ```

   before executing tests.  This builds the `sep_quantum` extension automatically.

2. Add or update tests to cover:

   * `Candle::fromJson` parsing numeric strings, mid/bid/ask payloads and ISO‑8601 timestamps.
   * Calibration and sweep scripts when `--use-native-quantum` is enabled.
   * End‑to‑end prime/ export scripts using a small mock dataset.

3. Run the test suite:

   ```bash
   pytest -q score/tests/test_logistics_features.py \
                score/tests/test_qfh_native.py
   ```

---

Following these steps will ensure you’re using real OANDA data end‑to‑end, that the Candle parsing fix is exercised, and that all plots and bridge metrics reflect the native QFH metrics.  Once the foundation paper is complete, you can use it as the baseline for your follow‑on papers.
