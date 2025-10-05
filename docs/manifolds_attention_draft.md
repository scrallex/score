# Manifolds Meet Attention — Draft

## Methods
- **Structural manifold features:** Every claim and evidence sentence carries O-space metrics — patternability, semantic alignment, coherence, stability, entropy, rupture density, and hazard \(\lambda\). These metrics drive the reliability head alongside the admission priors inherited from the manifold repairs. The feature bundle matches the signals defined in the twin-gating work and keeps the admit policy grounded in interpretable structure.
- **Transformer reliability head:** We reuse the lightweight encoder from *Attention Is All You Need* (Vaswani et al., 2017) with sinusoidal + phase channels and optional cross-attention into the evidence cache. The cross-attention block mirrors the citation alignment module sketched in 2509.13351v1, allowing the final hidden state to retrieve supporting spans before the reliability MLP emits admit probability and support margin.
- **Calibration hooks:** Training minimises binary cross-entropy on admit logits plus an \(\ell_2\) loss on the margin target. Validation wraps threshold sweeps (admit grid 0.1–0.9, margin grid −0.5–1.0) and Expected Calibration Error to keep probabilities auditable.

## Experiments
### FEVER
- Curriculum run (3:1 FEVER→SciFact batches, warm-started from the FEVER checkpoint) lands **test F1 0.748** (precision 0.691 / recall 0.817) with calibrated thresholds 0.2 / −0.5 (`results/experiments/fever_scifact_curriculum.json`). Baseline FEVER-only fine-tune remains at **test F1 ≈0.76**, so the mixed schedule trades ~1% absolute F1 for SciFact coverage while keeping Brier at 0.16.
- Admit probability mass stays broad (mean 0.27, std 0.32) and attention concentrates on a single evidence span (max weight mean 0.66), matching the behaviour logged in `results/analysis/attention_metrics.json`.

### SciFact
- **Zero-shot FEVER checkpoint:** validation F1 collapses to 0.0 with nearly all admit mass at <\(10^{-4}\) (`scifact_val_fever_init`), replicating the generalisation failure noted in the whitepaper demo.
- **Targeted fine-tune:** five SciFact epochs (batch 32, LR 5e-5) restore **val F1 0.58** / **test F1 0.52**; calibration pushes val F1 to 0.68 but still leaves the model under-confident (`results/experiments/scifact_finetune.json`).
- **Curriculum schedule:** the FEVER↔SciFact interleave lifts SciFact metrics to **val F1 0.69** / **test F1 0.64** with calibrated thresholds 0.3 / 1.0. Admit mass now peaks near 0.9 (mean 0.71) and attention entropy stays moderate (0.69), signalling that the model now commits when cross-domain evidence aligns (`results/experiments/scifact_val_curriculum_eval.json`).

### HoVer
- **Baseline (FEVER checkpoint):** HoVer validation F1 is 0.0, Brier 0.52, and 100% of admit probability lies in the first histogram bin; attention max weight averages 0.24, so the model refuses to pick supporting hops (`results/experiments/hover_val_fever_base_eval.json`).
- **Adapted run:** Three GPU epochs over the HoVer train split (3.2M steps, batch 48) nudge validation F1 to 0.095, drop Brier to 0.48, and lift mean admit probability to 0.048 with heavier mass beyond 0.2. Calibration (0.1 / −0.5) reaches F1 0.164. Attention remains diffuse (max weight 0.25) but entropy tightens slightly; see `results/experiments/hover_val_fever_adapt_eval.json`.
- **Artifacts:** calibration curves for HoVer, FEVER, and SciFact live in `results/eval/hover_dev/calibration_hover_adapt.png`, `results/eval/fever_dev/calibration_curriculum_fever.png`, and `results/eval/scifact_dev/calibration_curriculum_scifact.png`.

## Discussion
- **Attention behaviour:** Figure `docs/figures/attention_summary.png` captures the contrast: FEVER runs keep max attention ≈0.66 with mid-range entropy, SciFact gains sharper distributions post-curriculum (probability mean 0.71, entropy 0.69), and HoVer remains weakly focused even after fine-tuning. The diffuse HoVer pattern indicates we still miss multi-hop alignment.
- **Calibration impact:** HoVer improves from a flat reliability curve (ECE 0.52) to ECE 0.48, while FEVER’s curriculum model holds ECE at 0.12 and SciFact’s calibrated recall jumps to 0.91. Threshold sweeps surface workable operating points for each corpus; the combined calibration summary is recorded in `results/analysis/calibration_summary.json`.
- **Failure modes:** SciFact still exhibits overconfident false positives (Brier 0.34) and relies on high margins (threshold 1.0) to rein in recall. HoVer remains data-starved — attention entropy >1.45 and F1 <0.2 point to missing multi-hop evidence or insufficient memory coverage.

## Appendices
- **Ingestion pipelines:** `scripts/convert_fever_to_eval.py`, `scripts/convert_scifact_to_eval.py`, and `scripts/convert_hover_to_eval.py` emit STM-compatible `eval_detail` files with structural metrics, semantic hashes, and citations. Sentence segmentation for HoVer requires the bundled NLTK punkt models.
- **Dataset splits:** Deterministic IDs reside in `data/splits/fever_*`, `data/splits/scifact_*`, `data/splits/hover_train_*`, and `data/splits/hover_dev_*`. The curriculum run consumes `fever_train_dev/eval_detail.jsonl` and `scifact_train_dev/eval_detail.jsonl` with these splits.
- **Figure inventory:**
  1. `docs/figures/attention_summary.png` — aggregated attention/probability chart (updated).
  2. `results/eval/hover_dev/calibration_hover_adapt.png` — HoVer reliability diagram.
  3. `results/eval/fever_dev/calibration_curriculum_fever.png` — FEVER curriculum calibration.
  4. `results/eval/scifact_dev/calibration_curriculum_scifact.png` — SciFact curriculum calibration.
