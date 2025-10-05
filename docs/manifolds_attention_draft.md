# O-Space Reliability Meets Transformer Attention

## 1. Introduction: Problem Framing
- After re-running the Transformer gate with the relaxed GPU sweep, `whitepaper_demo` now reaches macro-F1 1.000 with 45.8% citation coverage (`results/eval/whitepaper_demo_transformer_relaxed/eval_summary.json`), demonstrating that the gate admits useful spans once evidence exists.
- `docs_demo` shows the same direction of travel—macro-F1 0.305, hallucination 0.875, citation coverage 0.083 (`results/eval/docs_demo_transformer_relaxed/eval_summary.json`)—but still lacks a dense pack, so most claims are declined.
- FEVER dev continues to emit nothing meaningful (macro-F1 0.166 vs. 0.567 baseline, citation coverage 0.0; `results/eval/fever_dev_transformer_relaxed/eval_summary.json`) because the truth-pack is empty. This keeps overall recall near zero despite relaxed thresholds.
- SciFact and HoVer remain stuck in the UNVERIFIABLE basin (`results/eval/scifact_transformer_relaxed/eval_summary.json`, `results/eval/hover_transformer_relaxed/eval_summary.json`), highlighting the twin problems of sparse evidence and domain mismatch.

| Pack | Baseline Macro-F1 | Transformer (relaxed) Macro-F1 | Relaxed Hallucination | Relaxed Citation Coverage |
| --- | ---: | ---: | ---: | ---: |
| FEVER dev | 0.567 | 0.166 | 0.814 | 0.000 |
| docs_demo | 0.570 | 0.305 | 0.875 | 0.083 |
| whitepaper_demo | 0.605 | 1.000 | 0.208 | 0.458 |
| SciFact dev | 0.554 | 0.154 | 0.167 | 0.000 |
| HoVer dev | 0.271 | 0.220 | 0.039 | 0.000 |

*Relaxed runs use dataset-specific sweeps: FEVER (`lambda_max=0.8`, `sigma_min=-0.5`, `structural=0.20`, `semantic=0.02`), SciFact/HoVer/docs_demo/whitepaper_demo (`lambda_max=0.6`, `sigma_min=-0.2`, `structural=0.25`, `semantic=0.05`); all retain `r_min=1`. Reliability margins stay at `p=0.05`, `margin=-0.5` unless noted.*

### Evidence Coverage Diagnostics (2025-10-05)
| Pack | Files in Pack | Bytes (MB) | Unique Strings | Notes |
| --- | ---: | ---: | ---: | --- |
| FEVER dev | 1,502 | 0.00 | 0 | Placeholder `.txt` files under `data/fever_pack/dev` leave the manifold empty. |
| SciFact dev | 283 | 0.44 | 7,893 | Covers only 5.5% of the 5,183-corpus; manifest paths point to ephemeral `/tmp/...` sources. |
| HoVer dev | 9,145 | 1.26 | 18,805 | Stores one gold sentence per page (9,145 pages), so no multi-hop context exists. |

The numbers above come from `analysis/truth_packs/*/manifold_state.json`. FEVER lacks any string metrics, SciFact retains partial abstracts without durable file paths, and HoVer’s pack mirrors the gold rationals without additional retrieval. These gaps explain the zero-recall outcomes on the public datasets and motivate the evidence-enrichment plan below.

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
- **E0 - Dataset bootstrapping:** FEVER ([fever.ai](https://fever.ai/)), SciFact ([researchgate.net](https://www.researchgate.net/publication/343901998_SciFact)), and HoVer ([arxiv.org/abs/2011.00685](https://arxiv.org/abs/2011.00685)) converters now emit STM-compatible `eval_detail.jsonl` files with structural metrics (`scripts/convert_*_to_eval.py`). Coverage audits still show whitepaper_demo lacks evidence density, explaining the zero-coverage sweep.
- **E1 - Final-answer scoring:** GPU retrains on FEVER reach test F1 0.756 with calibrated thresholds (checkpoint `models/reliability_fever_attn_full.pt`); whitepaper_demo remains unchanged because the reliability head is not yet wired into the admission path.
- **E2 - Margin/overlap calibration:** Calibration sweeps and temperature scaling reduce SciFact ECE from 0.207 to 0.075 (`results/analysis/scifact_temperature_finetune.json`), but we still need to propagate the calibrated thresholds into `reality_filter_eval.py`.
- **E3 - Phase encoding ablation:** Dropping phase channels cuts FEVER test F1 from 0.756 to 0.690; SciFact remains fragile without them, so the whitepaper will position phase information as a key ablation result.
- **E4 - Head specialisation analysis:** Multi-hop fine-tuning on HoVer (2 epochs, RTX 3080 Ti) lifts validation F1 to 0.727 and test F1 to 0.662 at calibrated thresholds 0.2 / 0.0 (`results/experiments/hover_multi_hop_trial.json`). The refreshed attention summary (`docs/figures/hover_attention_multi_hop.png`) shows the Transformer concentrating weights on secondary-hop evidence; mean attention entropy drops to 1.82 with 90th percentile max weight 0.20 (`results/experiments/hover_multi_hop_eval_summary.json`).
- **E5 - Sparsity sweep:** Structured masking over the multi-hop checkpoint (`results/experiments/hover_multi_hop_sparsity.json`) keeps admit precision/recall at 1.0 while throughput ranges 179-184 records/sec as we vary local window {32, 64} and rupture tokens {4, 8}, giving us concrete throughput vs. context-size trade-offs.

## 6. Integration Plan
- **Model implementation:** `src/sep_text_manifold/attn_ospace.py` hosts the Transformer backbone, phase fusion, cross-attention, and reliability head.
- **Training harness:** `scripts/train_reliability_attn.py` ingests eval details, trains with calibration sweeps, and logs admit precision/recall, Brier score, ECE, and attention entropy.
- **Dataset ingestion:** `scripts/convert_fever_to_eval.py`, `scripts/convert_scifact_to_eval.py`, and `scripts/convert_hover_to_eval.py` hydrate corpora into STM evaluation artefacts with structural metrics and citations.
- **Service swap:** `scripts/reality_filter_eval.py` now defaults to the Transformer gate (with `--disable-reliability` as an escape hatch); the outstanding work is rerunning FEVER/SciFact/HoVer packs on GPU hardware and wiring the refreshed metrics into the demos.
- **Logging & reports:** Attention artefacts are now timestamped (`results/eval/<pack>_transformer/attention_<timestamp>/`) and indexed in `results/attention_logs.txt`. After aggregation we collapse the per-claim heatmaps into mean-intensity figures (e.g. `docs/figures/attention_docs_demo_transformer_*.png`), then prune the raw PNGs so the repo stays lightweight. We still publish aggregate figures (`docs/figures/attention_summary.png`, `docs/figures/hover_attention_multi_hop.png`), while `scripts/plot_reliability_results.py` regenerates the comparison plots and writes the Markdown metrics table alongside `results/experiments/hover_multi_hop_eval_summary.json` for reproducibility.
- **CI guardrails:** `.github/workflows/attn-tests.yml` trains a 1-epoch demo model and ensures evaluator parity; next step is wiring the calibration comparison when the reality filter consumes the new admit scores.

## 7. Discussion and Open Questions
The CUDA reruns confirm the two headline hypotheses. First, the attention head is
decisive when sufficient evidence exists: FEVER reliability ablations still show
phase removal dropping test F1 from 0.756 to 0.690, and the feature-dimension
and MLP baselines trail the full Transformer despite sharing identical metrics.
Second, the gate now mirrors the evidence it sees. With the relaxed sweep the
transformer admits nearly every supported span on `whitepaper_demo` (macro-F1
1.000, citation coverage 0.458) while `docs_demo` remains recall-starved
(macro-F1 0.305, hallucination 0.875) because the pack still holds only a
handful of spans. FEVER continues to emit nothing under the relaxed settings,
underscoring that the dev pack needs actual text before we can claim recall
gains.

SciFact remains the weakest transfer story: even with a partial truth pack (283
of 5,183 abstracts) the relaxed gate still declines every claim, so macro-F1
stays at 0.154. HoVer mirrors that failure because the current pack stores only
the gold sentences; the FEVER-trained head has no multi-hop context to attend to,
so we either expand retrieval or frame the result explicitly as a negative
outcome that motivates domain-adaptive fine-tuning.

- **Evidence memory:** Should we pre-compute embeddings, learn them jointly, or keep a hybrid cache that fuses manifold metrics with token embeddings?
- **Attention sparsity vs coverage:** What balance between local restricted attention and global rupture tokens preserves admit precision while meeting the >=1k rps SLO?
- **API exposure:** Do we surface the reliability head as a standalone endpoint so planning/execution agents can query admit probabilities directly?
- **Auditability:** How do we version and store per-head attention maps so future audits can reconstruct decisions across FEVER, SciFact, HoVer, and internal packs?
- **Immediate actions:**
  1. Enrich the FEVER, SciFact, and HoVer truth packs: hydrate FEVER dev from `data/fever/wiki-pages`, ingest the full SciFact corpus (`external/scifact/data/corpus.jsonl`) plus lexical neighbours, and extend HoVer ingestion to cache full articles and multi-hop candidates.
  2. Calibrate or learn dataset-specific thresholds: run per-pack sweeps via `scripts/evaluate_reliability.py`/`train_reliability_attn.py --calibrate` and prototype a feature-conditioned admit/margin predictor.
  3. Domain-adapt the reliability head: fine-tune the Transformer gate on SciFact and HoVer (starting from the FEVER checkpoint) and explore curriculum mixes with `scripts/train_curriculum_fever_scifact.py`.
  4. Improve attention-map analysis: process targeted FEVER/SciFact claims with `scripts/aggregate_attention_snapshots.py` and emit per-head heatmaps for the manuscript.
  5. Keep the draft in sync: weave the coverage stats, calibration outcomes, and qualitative attention plots into `docs/manifolds_attention_draft.md` as they land.
  6. Plan the final GPU reruns: once packs and thresholds settle, regenerate metrics + figures, lock the manuscript, and document remaining limitations (evidence sparsity, domain shift, multi-hop retrieval).

## 8. Attention Map Retention
Every evaluation run writes its raw attention PNGs to
`results/eval/<pack>_transformer/attention_<timestamp>/`. The aggregator script
(`scripts/aggregate_attention_snapshots.py`) collapses those hundreds of tiles
into a single mean-intensity chart per run and appends the aggregate path to
`results/attention_logs.txt`. The companion plotting script regenerates the
publication figures from the reliability summaries; together they ensure that a
reviewer can reproduce the visuals by running:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/reality_filter_eval.py ...
python scripts/aggregate_attention_snapshots.py
python scripts/plot_reliability_results.py --device cuda
```

The appendix will reference both scripts so future readers know how to recover
the raw attention grids if they need to audit a specific claim.

## 9. Conclusions and Limitations
Transformer gating improves reliability metrics only when the evidence manifold
is rich enough to support admissions. Phase channels remain critical—the FEVER
ablation confirms a ~0.07 F1 drop without them—while HoVer requires explicit
multi-hop retrieval to surface supporting spans. Sparse packs (docs_demo,
whitepaper_demo) and missing manifolds (SciFact, HoVer) are the dominant
failure modes: the gate simply declines everything. Future work should focus on
ingesting those corpora, relaxing structural thresholds once real citations are
available, and exploring domain-adaptive fine-tuning so SciFact and HoVer
benefit from the same calibrated gate that helps FEVER.
