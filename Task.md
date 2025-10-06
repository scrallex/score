# Whitepaper Outline: O-Space Reliability Meets Transformer Attention

The goal of this whitepaper is to merge our O-space reliability gating work with the Transformer attention paradigm of *Attention Is All You Need* (Vaswani et al., 2017), demonstrating how manifold-derived structural signals can be standardised within an attention-backed admit pipeline. Below is the working outline, with narrative goals and concrete implementation hooks for each section.

## 1. Introduction: Problem Framing
- Restate the operational hazards already observed: hallucination rate stuck at 1.0, token-support passes on zero-margin URI hits, and brittle post-processing in `reality_filter_eval.py`.
- Motivate “reliability-gated admission” for both generated answers and market signals: we need to admit spans/candles only when structural evidence exists in the O-space memory.
- Introduce O-space as the content-addressable manifold built from QFH/QBSA metrics (coherence, stability, entropy, rupture, hazard) and emphasise the need to replace heuristic gates with a learnable admit policy.
- Preview how Transformer attention gives a principled mechanism for global lookups, multi-relational reasoning, and evidence alignment.

## 2. Primer on Transformer Attention
- **Scaled dot-product attention (Sec. 3.2.1, Fig. 2).** Summarise queries/keys/values, the √dₖ scaling, and softmax weighting; explain that this is the same content-addressable behaviour we want for O-space retrieval.
- **Multi-head diversity (Sec. 3.2.2).** Describe how projecting into multiple subspaces lets different heads specialise (support, contradiction, recurrence, noise). Connect this to the reliability bands we currently maintain manually.
- **Long-range connectivity (Table 1).** Highlight O(1) maximum path length and sequential operations for self-attention versus O(n) for RNNs, justifying why attention is ideal for rupture-spanning support discovery.
- **Position and phase (Sec. 3.5).** Detail sinusoidal encodings and propose extending them with our prime/phase ticks so rhythmic structure is available to the model without losing extrapolation capacity.
- **Cross-attention for evidence gating (Sec. 3.2.3).** Map decoder-to-encoder attention onto our “query span attends to manifold memory” requirement and foreshadow a cross-attention block that learns citation alignment.

## 3. Recap of the Manifold / QFH / QBSA Framework
- Summarise the metrics (coherence, stability, entropy, rupture density, hazard λ) and how they yield signatures for windows across text and price streams.
- Describe the manifold construction pipeline: sliding window ingestion, repetition counting, hazard thresholds, and the existing ORJSON manifests.
- Review the twin-repair loop and semantic gating currently used, noting the failure modes identified in recent commits (latched hallucination flag, unchecked URI override).
- Reference existing documentation (`QFH_Manifold_Foundation.tex`, `reliability_gated_recurrence_polished.tex`) to show prior empirical grounding.

## 4. Proposed Model Architecture (O-Space Transformer)
1. **Backbone:** Lightweight Transformer (2–4 layers, 4–8 heads) with causal self-attention over the span/candle sequence; outputs contextual embeddings aimed at predicting reliability metrics.
2. **Evidence memory + cross-attention:** Maintain a key/value cache of truth-pack strings or historical price windows; implement a cross-attention layer where the current query attends over this memory to retrieve supportive evidence.
3. **Reliability head:** MLP on the final hidden state producing both an admit probability and a calibrated support margin; trained on final adjudicated labels (after repairs) so metrics reflect end-state decisions.
4. **Phase/prime encoding:** Concatenate standard sinusoidal encodings with prime/beat channels to inject O-space phase information while keeping compatibility with the Transformer positional scheme.
5. **Structured sparsity:** Employ restricted self-attention (local window plus global rupture tokens) to preserve throughput (target ≥1k rps) while retaining O(1) access to critical anchors.

## 5. Experimental Programme
- **E0 – Dataset bootstrapping:** Build the neutral example pack from `data/corpus_example/` with `scripts/reality_filter_pack.py` so the SRI quick start mirrors the repository defaults.
- **E1 – Final-answer scoring:** Train/evaluate on repaired answers; report macro-F1, admit precision/recall, calibration error. Expect hallucination rate to drop versus heuristic pipeline.
- **E2 – Margin/overlap calibration:** Impose dual thresholds on admit probability and attention mass over evidence; produce reliability diagrams to confirm zero-margin URI hits no longer slip through.
- **E3 – Phase encoding ablation:** Compare models with/without extra phase channels; track long-lag recurrence recall and head specialisation metrics.
- **E4 – Head specialisation analysis:** Probe attention heads for support/contradiction/recurrence/noise roles, mirroring qualitative attention visualisations from the original paper.
- **E5 – Sparsity sweep:** Vary local window size and number of rupture tokens; ensure throughput remains within SLOs while monitoring admit precision.

## 6. Integration Plan
- **Model implementation:** Add `src/sep_text_manifold/attn_ospace.py` containing the Transformer backbone, evidence cross-attention, and reliability head.
- **Training harness:** Create `scripts/train_reliability_attn.py` to ingest evaluation JSONL, train the model, and log precision/recall, calibration error, and attention entropy.
- **Dataset ingestion:** Point contributors to `archive/datasets/fever/` for the deprecated converters and keep new documentation focused on the SRI/SBI flow.
- **Service swap:** Modify `scripts/reality_filter_eval.py` to consume the trained reliability head output instead of heuristic token-support overrides; thresholds become calibrated probabilities.
- **Logging & reports:** Persist per-head attention maps under `results/eval/<pack>/attention_heads/` and surface them in the report markdown to show which evidence drove each admit.
- **CI guardrails:** Extend tests to compare confusion matrices and calibration metrics between detail outputs and summaries; fail if divergence exceeds tolerated bounds.

## 7. Discussion and Open Questions
- How should the evidence memory be initialised—precomputed embeddings, joint training, or hybrid KV caches?
- What is the optimal balance between local restricted attention and global rupture tokens for rare long-range dependencies?
- Should the reliability head be exposed as an independent API so planning/execution services can query admit probabilities directly?
- How do we ensure reproducibility and versioning for attention maps so audits can trace decisions post-hoc?

This outline sets the structure for the whitepaper and the accompanying implementation work. Next steps: build the `attn_ospace` module, stand up the training harness, and begin running E1/E2 experiments to validate the Transformer-backed admit pipeline.

## Experiment log (2025-10-04)
- **Whitepaper demo (training, 5 epochs, CPU):** final train loss 0.051, val loss 0.035, val Brier 0.0011; checkpoint stored at `models/reliability_whitepaper_demo.pt` with summary `results/experiments/whitepaper_demo_training_summary.json`.
- **Docs demo (training, 5 epochs, CPU):** final train loss 0.097, val loss 0.224, val Brier 0.0488; checkpoint stored at `models/reliability_docs_demo.pt` with summary `results/experiments/docs_demo_training_summary.json`.
- **Evaluation baselines vs. reliability:**
  * `results/experiments/whitepaper_demo_baseline` vs `whitepaper_demo_reliability` — reliability model currently leaves macro-F1 (0.115) and hallucination rate (1.0) unchanged, indicating token-support paths are not yet firing on this small pack.
  * `results/experiments/docs_demo_baseline` vs `docs_demo_reliability` — same outcome; reliability gating has no effect pending richer evidence hooks.
- **CI update:** `.github/workflows/attn-tests.yml` now trains a 1-epoch demo model and runs the evaluator against it, in addition to the reliability unit tests.

## Experiment log (2025-10-05)
- **Legacy ingestion tooling:** `archive/datasets/fever/convert.py` and its siblings remain runnable for archival reproductions but should not resurface in docs, automation, or quick-start snippets.
- **Reliability pipeline refresh:** `scripts/train_reliability_attn.py` now emits calibration sweeps (`--calibrate-thresholds`), Expected Calibration Error, and richer evidence encodings; `scripts/reality_filter_eval.py` records transformer probabilities/margins per repaired span.
- **Reliability gate (optional):** Any classifier heads calibrated against FEVER-era checkpoints should be referenced as legacy assets; future tuning should document corpus-neutral sweeps in `results/sbi/REPORT.md` (or a dedicated appendix).

## Experiment log (2025-10-06)
- **HoVer multi-hop retrieval trial:** Augmented train/dev eval_detail files with top-2 TF-IDF hops and fine-tuned the reliability head for 2 epochs on the RTX 3080 Ti (batch 32, LR 5e-5, `attention_entropy_weight=1e-3`). Validation F1 reached 0.727, test F1 0.662 at calibrated thresholds 0.2 / 0.0; checkpoint stored at `models/reliability_hover_multi_hop.pt` with metrics in `results/experiments/hover_multi_hop_trial.json` and calibration plot at `results/eval/hover_dev_multi_hop/calibration_hover_multi_hop.png`.
- **Attention aggregation:** Collapsed per-claim attention PNGs into a single figure (`docs/figures/hover_attention_multi_hop.png`) that highlights the highest-probability claims; evaluation summary (`results/experiments/hover_multi_hop_eval_summary.json`) reports mean attention entropy 1.82 and 90th-percentile max weight 0.20.
- **Structured sparsity sweep:** Replayed the multi-hop checkpoint with local window {32, 64} and rupture tokens {4, 8}; precision/recall remained 1.0 while throughput held between 179-184 rec/s (`results/experiments/hover_multi_hop_sparsity.json`).

## Immediate Actions
1. Extend the HoVer multi-hop trial only if it can be demonstrated without reintroducing FEVER dependencies; otherwise keep the curriculum scripts quarantined in `archive/datasets/fever/`.
2. Apply temperature scaling to the SciFact curriculum checkpoint (follow `docs/calibration_checklist.md`) to rein in over-confident admits and lock calibrated thresholds.
3. Integrate `scripts/evaluate_reliability.py` into CI once a corpus-neutral calibration path exists (the FEVER/SciFact figures are now archived).
