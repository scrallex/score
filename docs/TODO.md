do this

Here’s a concise checklist for getting your environment and analysis back on track now that the SEP trading stack is live:

1. **Investigate the JSON type error** – The `prime` logs show repeated `nlohmann::json` type errors before and after storing manifolds. These usually come from writing a non‑numeric field as a number or vice‑versa. Inspect the C++ manifold-writing code and any custom serialization to ensure all floating‑point values are written as doubles rather than integers (e.g. cast values to `double` before serializing). Once fixed, re‑run the prime job and confirm that no exceptions occur.

2. **Verify API endpoints** – With the containers running, test key endpoints to ensure data is accessible:

   * `curl http://localhost:8000/api/manifold/latest?instrument=EUR_USD` should return the most recent manifold with fields like `coherence`, `stability`, `entropy`, `rupture`, `lambda`, and `repetition`.
   * `curl http://localhost:8000/api/opt/slice?instrument=EUR_USD&window_min=120` should return optimisation slices.
     These APIs should match the definitions documented in your code and whitepaper.

3. **Populate FX snapshots** – Now that Valkey is reachable, run:

   ```bash
   python scripts/ops/prime_qfh_history.py --instrument EUR_USD --days 30 --store-manifold-to-valkey
   python scripts/ops/export_manifold_snapshots.py --instrument EUR_USD --minutes 43200 --out output/manifolds_native/EUR_USD_snapshots.csv
   ```

   Repeat for other pairs (USD\_JPY, GBP\_USD, etc.). These CSVs are needed for the live‑FX figures.

4. **Regenerate warmup datasets** – The figure script expects data under `output/warmup/…`. Ensure the prime job writes JSON files there (e.g. `output/warmup/EUR_USD/2025-09-18.json`). If the path differs, either adjust `plot_whitepaper_figures.py` to accept your actual directories or symlink your current outputs into `output/warmup`.

5. **Rebuild figures and tables** – Once FX snapshots exist, run:

   ```bash
   python scripts/plot_whitepaper_figures.py \
     --synthetic results/qfh_synthetic_native.json \
     --planbench output/planbench_by_domain/logistics/gold_state_logistics_native.json \
     --fx output/manifolds_native/EUR_USD_snapshots.csv \
     --bridge docs/note/bridge_metrics.json
   ```

   This will generate histograms of event types, metric distributions and updated bridge summaries. Verify that the script now produces all plots without errors.

6. **Finish the QFH foundation paper** – Using the fresh figures and tables, complete `QFH_Manifold_Foundation.tex`. Make sure to:

   * Describe how native metrics are incorporated into logistic features and how the `native_metrics_provider` handles the QFH engine.
   * Include synthetic, PlanBench and live‑FX results, along with bridge correlations.
   * Add a reproducibility appendix with the commands used in steps 3–5.

7. **Update CI and tests** – Ensure your CI pipeline runs `pip install .[native]` before `pytest` so the pybind module is built automatically. Add tests to exercise the new calibration and sweep scripts with the `--use-native-quantum` flag. These tests should verify that logistic features include native fields (coherence, stability, entropy, rupture, lambda, signature).

8. **Plan next papers** – Once the foundation paper is complete and the live data pipeline is stable, you can use it as the authoritative reference when splitting the remaining work into the PlanBench‑only paper and the combined STM↔spt paper.

By following these steps you’ll have a running trading stack, corrected manifold exports, robust live datasets for your figures, and a completed foundation paper that cleanly documents the QFH/QBSA manifold technology.