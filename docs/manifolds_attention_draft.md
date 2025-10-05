# O-Space Reliability Meets Transformer Attention

## 1. Introduction: Problem Framing
- Whitepaper demo runs still rely on the heuristic reality filter: macro-F1 sits at 0.115 with hallucination rate pinned at 1.0 (`results/eval/whitepaper_demo/eval_summary.json`), even though the report summary shows 3 approved spans out of 5 (`results/report/whitepaper_demo.md`).
- Threshold sweeps collapse to zero coverage because the heuristic admit gate deactivates every span once the structural margin falls below 0.46, leaving us with no calibrated operating point (`results/eval/whitepaper_demo/best_thresholds.json`).
- URI hits continue to bypass structural checks: zero-margin links emit ``SUPPORTED`` in the baseline answers while the repaired answers refuse to admit, highlighting the brittleness of the current token-support override.
- The goal is to replace the rule set with an attention-backed admit policy that learns when structural evidence in O-space is present, so we can gate both generated answers and market spans on measurable support.

## 2. Primer on Transformer Attention
- **Scaled dot-product attention (Sec. 3.2.1, Fig. 2, Vaswani et al., 2017):** Queries attend to keys via sqrt(d_k) scaling and softmax weighting, mirroring our O-space lookups where a span retrieves manifold neighbours based on structural similarity.
- **Multi-head diversity (Sec. 3.2.2):** Splitting into 4-8 heads lets each projection specialise (support, contradiction, recurrence, noise suppression) and replaces the manually maintained reliability bands in the current filter.
- **Long-range connectivity (Table 1):** Self-attention delivers $O(1)$ maximum path length versus the $O(n)$ latency of the recurrent twin-repair loop, which is essential for rupture-spanning evidence.
- **Position and phase (Sec. 3.5):** Standard sinusoidal encodings preserve generalisation; we add prime/phase channels so rhythmic structure enters the network without losing extrapolation ability.
- **Cross-attention for evidence gating (Sec. 3.2.3):** Decoder-to-encoder attention maps directly onto "query span attends to manifold memory," foreshadowing a cross-attention block that grounds admit decisions in stored citations or price windows.

## 3. Recap of the Manifold / QFH / QBSA Framework
- Every window carries coherence, stability, entropy, rupture density, and hazard lambda plus derived patternability and semantic metrics; these live in the ORJSON manifolds emitted by the QFH/QBSA stack.
- Sliding-window ingestion feeds `sep_text_manifold.encode_window`, repetition counts, and hazard thresholds into manifests that drive both the heuristic guardrail and the new transformer features.
- The twin-repair loop still runs end to end but exhibits failure modes documented in recent runs: latched hallucination flags and unchecked URI overrides inside `scripts/reality_filter_eval.py`.
- Prior whitepapers (`docs/whitepaper/QFH_Manifold_Foundation.tex`, `docs/whitepaper/reliability_gated_recurrence_polished.tex`) capture empirical support for the manifold, giving us a foundation to cite in the new draft.

## 4. Proposed Model Architecture (O-Space Transformer)
1. **Backbone:** Lightweight Transformer (2-4 layers, 4-8 heads) with causal self-attention over span/candle sequences, tuned via `sep_text_manifold.attn_ospace.OspaceTransformer`.
2. **Evidence memory + cross-attention:** Truth-pack strings or price windows flow through the `EvidenceEncoder`, yielding the key/value cache consumed by the cross-attention block.
3. **Reliability head:** A two-branch MLP outputs admit probability and support margin, trained on repaired labels pulled from evaluation artefacts.
4. **Phase/prime encoding:** Phase tensors align with prime ticks and rhythm metrics, concatenated with sinusoidal encodings so the model sees O-space beat structure.
5. **Structured sparsity:** Local-plus-rupture attention masks keep throughput at the >=1k rps target while retaining global access to hazard spikes.
6. **Calibration hooks:** Admit/margin thresholds and optional temperature scaling are baked into `scripts/train_reliability_attn.py` so the model emits calibrated probabilities.

## 5. Experimental Programme
- **E0 - Dataset bootstrapping:** FEVER, SciFact, and HoVer converters now emit STM-compatible `eval_detail.jsonl` files with structural metrics (`scripts/convert_*_to_eval.py`). Coverage audits still show whitepaper_demo lacks evidence density, explaining the zero-coverage sweep.
- **E1 - Final-answer scoring:** GPU retrains on FEVER reach test F1 0.756 with calibrated thresholds (checkpoint `models/reliability_fever_attn_full.pt`); whitepaper_demo remains unchanged because the reliability head is not yet wired into the admission path.
- **E2 - Margin/overlap calibration:** Calibration sweeps and temperature scaling reduce SciFact ECE from 0.207 to 0.075 (`results/analysis/scifact_temperature_finetune.json`), but we still need to propagate the calibrated thresholds into `reality_filter_eval.py`.
- **E3 - Phase encoding ablation:** Dropping phase channels cuts FEVER test F1 from 0.756 to 0.690; SciFact remains fragile without them, so the whitepaper will position phase information as a key ablation result.
- **E4 - Head specialisation analysis:** `docs/figures/attention_summary.png` visualises head entropy and max weights across FEVER, SciFact, and HoVer; HoVer stays diffuse (mean max ~=0.25), motivating multi-hop retrieval.
- **E5 - Sparsity sweep:** Structured masks are still pending; we hold >=1k rps in the existing demo but need to formalise throughput + admit precision trade-offs.

## 6. Integration Plan
- **Model implementation:** `src/sep_text_manifold/attn_ospace.py` hosts the Transformer backbone, phase fusion, cross-attention, and reliability head.
- **Training harness:** `scripts/train_reliability_attn.py` ingests eval details, trains with calibration sweeps, and logs admit precision/recall, Brier score, ECE, and attention entropy.
- **Dataset ingestion:** `scripts/convert_fever_to_eval.py`, `scripts/convert_scifact_to_eval.py`, and `scripts/convert_hover_to_eval.py` hydrate corpora into STM evaluation artefacts with structural metrics and citations.
- **Service swap:** `scripts/reality_filter_eval.py` now accepts a reliability checkpoint flag, but the whitepaper demo still defaults to heuristics; we need to switch the admit path to the Transformer output before rerunning metrics.
- **Logging & reports:** Attention maps for FEVER/SciFact/HoVer runs live in `results/eval/fever_attention/`, `results/eval/fever_scifact_attention/`, and related directories; the aggregate figure (`docs/figures/attention_summary.png`) is ready for the whitepaper appendix.
- **CI guardrails:** `.github/workflows/attn-tests.yml` trains a 1-epoch demo model and ensures evaluator parity; next step is wiring the calibration comparison when the reality filter consumes the new admit scores.

## 7. Discussion and Open Questions
- **Evidence memory:** Should we pre-compute embeddings, learn them jointly, or keep a hybrid cache that fuses manifold metrics with token embeddings?
- **Attention sparsity vs coverage:** What balance between local restricted attention and global rupture tokens preserves admit precision while meeting the >=1k rps SLO?
- **API exposure:** Do we surface the reliability head as a standalone endpoint so planning/execution agents can query admit probabilities directly?
- **Auditability:** How do we version and store per-head attention maps so future audits can reconstruct decisions across FEVER, SciFact, HoVer, and internal packs?
- **Immediate actions:**
  1. Prototype multi-hop retrieval or expanded evidence memory ahead of the HoVer adaptation run to reduce attention entropy and lift admit mass.
  2. Apply SciFact temperature scaling inside the evaluation pipeline so calibrated thresholds govern the admit/repair path.
  3. Integrate `scripts/evaluate_reliability.py` into CI to regenerate calibration plots and histograms for FEVER, SciFact, HoVer, and whitepaper packs.
