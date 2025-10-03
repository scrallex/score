# Manifold Coherence for Hallucination Control

## Problem Statement
Large language models fabricate citations, misstate figures, or synthesize policies that never existed. Traditional hallucination filters focus on lexical overlap or rule-based static checks, leaving the model free to produce novel-yet-false statements that look plausible. The structural manifold gives us a quantitative medium for saying "this response was seen before under similar conditions" and a way to refuse outputs that do not echo trusted precedent. Our goal is to formalize how STM/QFH coherence becomes a hallucination gate and how other systems can embed the medium.

## Terminology Refresh
- **Structural manifold (STM/QFH)**: the five-metric space (coherence `c`, stability `s`, entropy `H`, rupture `rho`, hazard `lambda`) derived from byte windows. Each window is assigned a discrete signature, and repetition counts determine whether the state has a reliable precedent.
- **Patternability**: the derived structural rhythm we plot on the X-axis of the demo scatter; it blends coherence, stability, and hazard to measure how repeatable a window is.
- **Semantic bridge**: transformer embeddings that project each window into cosine similarity against task-specific seeds. The bridge links structural rhythm to intent.
- **Manifold medium**: a materialized store of structural signatures, semantic fingerprints, and repetition statistics for a bounded corpus or online stream.
- **Coherence (co-here-ence)**: a targeted query for previously observed structural signatures. A candidate output is coherent when its signature has occurred enough times within the target manifold and the associated hazard is below threshold.
- **Target manifold**: the subset of the medium scoped to the question, policy domain, or dataset we are willing to treat as ground truth. It may be built on-the-fly from retrieval results, a curated knowledge pack, or a session transcript.

## Hallucination Control Strategy
1. **Context selection** – When the user issues a question, build a target manifold by ingesting the authoritative corpus (documents, telemetry, prior decisions) relevant to the query.
2. **Prompt execution** – Let the LLM draft responses, but stream each candidate sentence, claim, or table row into the manifold encoder just like source documents.
3. **Coherence verification** – For every candidate span:
  - Encode its structural signature and compare to the target manifold via `/stm/seen`.
  - Require `repetitions >= r_min` within a sliding lookback (e.g., 60 minutes for telemetry or the entire curated pack for docs).
  - Enforce `lambda <= lambda_max` and optionally stability/coherence floors to reject structurally erratic echoes.
  - Blend semantic similarity: demand cosine similarity >= `sigma_min` with the seed centroid representing the topic.
  - The admission rule can be expressed as

    ```
    admit(x) = [repetitions(x) >= r_min] and [lambda(x) <= lambda_max] and [sigma(x) >= sigma_min].
    ```
4. **Decision layer** – Responses that fail coherence are routed to repair: either prompt the LLM for sources, retrieve nearby twins as evidence, or directly decline with an explanation of missing precedent. Responses that pass coherence are annotated with the matched twins and returned.

The combination mirrors the reliability-gated recurrence detector: hallucination safety is treated as an admission problem. Outputs are admissible only when they materially co-here with trusted structure.

## Interaction Conditions
To make the gate actionable, we define interaction contracts:
- **Signature match**: The system must locate at least one twin within distance `epsilon` in signature space (structural) and embeddings (semantic).
- **Repetition quorum**: Twins must exhibit `repetitions >= r_min`. For conversational settings `r_min=2` may suffice; compliance/legal corpora may set `r_min >= 3`.
- **Hazard gate**: Accept only when `lambda <= lambda_max`. High hazard indicates the manifold is structurally stressed; treat this as an instability warning and block novel claims.
- **Semantic alignment**: A candidate must align with a topic seed pack. For hallucination vetting we often maintain dual seed families: factual claims (e.g., "regulation", "statute", "clinical trial") and disallowed novelty ("speculation", "rumor"). The guardrail approves only when it is close to the factual centroid and far from the novelty centroid.
- **Evidence bundle**: Each approved response should ship with the identity of matched windows (document URI, timestamp, or citation) so downstream consumers can audit.

## Architectural View
```
[Authoritative Corpus] --> [STM/QFH ingest] -- creates --> [Target Manifold Store]
                                        \                               |
                                         \                             v
                                          --> [Prompted Response Encoder] --(signatures)--> [Coherence Gate]
                                                                                                 |
                                         [Seed Pack] --(embeddings)--> [Semantic Bridge] -------/
```
- **Target manifold store**: Backed by Valkey or files, carries signatures, repetition counts, and metric slices.
- **Coherence gate**: Implements admission policy (`r_min`, `lambda_max`, `c_min`).
- **Semantic bridge**: Provides seed similarity and distance-to-novelty classifiers.
- **Response encoder**: Streams generated spans into the same STM pipeline to ensure symmetry with corpus ingestion.
- **Repair loop**: When the gate rejects a span, the orchestrator can request citations, shrink the prompt to a verified subset, or fall back to extractive answers from the matched twins.

## Implementation Hooks
- **Ingestion**: `stm ingest` with `--store-signals --window-len` tuned to the granularity of the knowledge domain. Enable `--keep-ids` so responses can cite canonical references.
- **Seen service**: `/stm/seen` (or `SeenEngine`) becomes the primary API. Configure it to accept mixed structural + embedding criteria.
- **Guarded generation**: Wrap the LLM in a streaming callback that intercepts sentences, encodes them with QFH metrics, and waits for the gate decision before emitting tokens.
- **Dashboarding**: Extend the semantic guardrail scatter plot to highlight WebUI claims: green for coherent, amber for low repetitions, red for hazard violations.
- **Telemetry**: Persist adjudication decisions (`approved`, `blocked`, `repair`) and the twin IDs. Feed these back into training data for active learning.

## Usage Patterns
- **Inline guardrail**: Use the gate in real time while the model is responding. Enforce strict coherence for regulated workloads (finance, healthcare).
- **Post-hoc adjudication**: Batch process generated reports, flag sections with low repetitions or high hazard, and route them to human reviewers.
- **Retrieval augmentation**: Instead of rejecting hallucinated spans, the gate triggers a retrieval step to inject coherent snippets into the prompt, reshaping the model toward reproducible answers.
- **Federated knowledge packs**: Maintain multiple manifolds (operations manual, case law, telemetry). The orchestrator selects the pack whose coherence matches the question intent.

## Research Extensions
- **Adaptive thresholds**: Calibrate `lambda_max` and repetition counts per corpus using permutation sweeps, similar to the logistics studies in the STM whitepaper.
- **Confidence scoring**: Combine coherence, hazard residual, and semantic margin into a probability of correctness. Feed this into downstream ranking or user interfaces.
- **Contrastive training**: Fine-tune smaller models to predict coherence directly from text spans, distilling the manifold into a lightweight classifier.
- **Temporal drift handling**: Monitor when previously coherent spans drop below `r_min` as data evolves. Use this drift to signal when knowledge packs need curation.

## Immediate Next Steps
1. Prototype the guarded generation loop using `SeenEngine` and the existing semantic bridge seeds to vet hallucinations in the semantic_guardrail_stream demo.
2. Backfill a regulatory knowledge pack, build its target manifold, and pilot the coherence gate on mock compliance Q&A.
3. Instrument decision telemetry and scatter dashboards so operators can visualize the ratio of coherent vs. blocked spans in real time.
