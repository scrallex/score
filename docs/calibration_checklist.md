# Calibration Checklist (Supplemental)

1. **Gather predictions:**
   - Run `scripts/evaluate_reliability.py <eval_detail.jsonl> <checkpoint>` with `--output-json` and dataset-specific `--index` to export admit probabilities, margins, and labels.
   - Verify the histogram in the JSON output no longer collapses into the first bin; if it does, rerun training or seed from a warmer checkpoint.
2. **Threshold sweep:**
   - Use the same command with `--calibrate` (and optional `--admit-grid` / `--margin-grid`) to record the F1 frontier.
   - Copy the best admit / margin thresholds into the experiment summary (`results/analysis/calibration_summary.json`).
3. **Reliability diagram:**
   - Supply `--calibration-plot PATH.png` to capture the reliability curve. Confirm the curve tracks the diagonal and note any low-probability skew.
4. **Temperature scaling (optional):**
   - Run `scripts/calibrate_temperature.py` with the validation split to optimise the temperature parameter.
   - Re-evaluate with the scaled checkpoint and update Brier/ECE metrics in the supplemental table.
5. **Document artefacts:**
   - Store calibration PNGs under `results/eval/<corpus>/` and reference them alongside the summary metrics in the appendix.
   - Add the thresholds and scaling constants to the release checklist before shipping a new reliability head.
