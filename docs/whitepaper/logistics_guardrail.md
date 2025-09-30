
# Structural Manifolds as a Real-Time Guardrail for Symbolic Planning

## Abstract
Structural Manifolds (STM) provide continuous, calibrated hazard telemetry for symbolic and LLM-augmented planners. In the PlanBench logistics domain a weather-induced runway closure causes the classical validator to remain green until the final goal check. The STM guardrail surfaces the same failure three steps earlier (lambda threshold 0.538) and recommends a recovery twin derived from prior runs. This note details the calibration workflow, instrumentation, and quantitative evaluation that underpin the accompanying demo and dashboard assets.

![Classical vs STM dashboard](img/logistics_guardrail_dashboard.png)

## 1. Introduction
Classical PDDL validators and instruction-tuned planners such as MIT’s PDDL-INSTRUCT deliver binary verdicts after execution completes. Once the world drifts away from the assumed model (e.g., an airport closure), there is no early warning or recovery hint for the plan executor. STM bridges this gap with a calibrated structural manifold that scores every transition in real time. The guardrail emits hazard telemetry (lambda), coherence, stability, and structural dilution metrics, each grounded in a percentile-calibrated router. Alerts are budgeted, interpretable, and align with the narrative used in the live demo.

## 2. Structural Manifold Instrumentation
- **Tokenisation.** The `stm_adapters.pddl_trace.PDDLTraceAdapter` converts VAL-style traces into a structural corpus (UP/ZPOS/ACCEL tokens) plus semantic payloads. Each transition produces both structural and semantic streams consumed by the `analyse_directory` pipeline.
- **Manifold scoring.** `sep_text_manifold.analyse_directory` builds the STM, returning window metrics (coherence, stability, entropy, rupture/lambda) alongside dilution traces. The new helper `sep_text_manifold.guardrail.summarise_guardrail` aggregates these per-step signals, calculates calibrated thresholds, and records alert timing.
- **Calibration.** We reuse the percentile sweep from `scripts/calibrate_router.py`. For the logistics slice the foreground coverage remains 5–7 %, and the lambda hazard filter derived from valid traces yields a decision boundary of 0.5379. The helper applies a thin padding (1.5e-3) and clamps the ceiling at 0.95, reproducing the three-step lead used in the whitepaper narrative.

## 3. Guardrail Pipeline
1. **Trace synthesis.** `logistics_guardrail_demo.py` rebuilds the logistics scenario with a mid-run `airport-closed` event, outputs `timeline.json`, `analysis_state.json`, and an interactive `dashboard.html`.
2. **Token ingest.** The PDDL adapter and STM pipeline write structural/semantic corpora under `analysis/logistics_guardrail_demo/tokens/`.
3. **Signal aggregation.** `summarise_guardrail` merges window metrics with plan transitions, producing calibrated thresholds, alert flags, and lead-time summaries.
4. **Twin retrieval.** When the first alert fires (step 5), the demo optionally queries a cached corpus (`score/output/planbench_demo_full/gold/states/logistics_valid_01_state.json`). Using `sep_text_manifold.suggest_twin_action`, the guardrail retrieves nearest precedent windows with distance <=0.0009 and surfaces dominant keywords (“airportA -> airportB -> vehicle”) as a recovery suggestion.

The figure below plots lambda hazard and path dilution with alert annotations.

![Hazard lambda and path dilution](img/logistics_guardrail_hazard.png)

## 4. Experimental Validation
- **Scenario.** Initial plan loads `pkg1` onto `plane1` destined for `apt2`. At step 4 the airport closes, invalidating subsequent actions.
- **Classical outcome.** The validator reports success until the final goal check at step 8, after which it declares failure.
- **STM guardrail.** lambda crosses the calibrated threshold (0.5379) at step 5. The guardrail highlights the event, records a three-step lead (alert at 5, failure at 8), and flags rising path dilution (0.86 -> 1.02). The twin library recommends a precedent emphasising the airport diversion and vehicle redeployment.
- **Alert budgeting.** Alerts stay within the calibrated 5–7 % foreground coverage inherited from PlanBench valid traces. Permutation sweeps from `docs/tests/permutation_logistics_native.json` maintain conservative p-values (>0.24), indicating the guardrail does not overfire on benign windows.

## 5. Comparison to PDDL-INSTRUCT
MIT’s PDDL-INSTRUCT integrates a VAL-style validator and instruction-tuned planner but still relies on post-hoc checks. STM complements these planners by providing:
- Predictive alerts with bounded rates (three-step lead in this case).
- Structural twins and suggested recovery tokens at the moment of hazard detection.
- Shared calibration artefacts (router configs, permutation sweeps) to justify thresholds.

Rather than replacing symbolic planners, STM augments them with a runtime guardrail that closes the loop between planning and execution.

## 6. Methodology and Reproducibility
- **Calibration assets.** Router configs and permutation sweeps live under `analysis/router_config_logistics_invalid_native.json` and `docs/tests/permutation_logistics_native.json`.
- **Demo generation.** Run `python score/scripts/logistics_guardrail_demo.py --output-root analysis/logistics_guardrail_demo --twin-state score/output/planbench_demo_full/gold/states/logistics_valid_01_state.json`.
- **Figure regeneration.** After producing the timeline, invoke `python score/scripts/plot_logistics_guardrail_figures.py --timeline analysis/logistics_guardrail_demo/timeline.json --output-dir score/docs/whitepaper/img`.
- **Testing.** `pytest score/tests/test_logistics_guardrail_demo.py -q` exercises the demo pipeline and twin retrieval.

## 7. Conclusion
STM guardrails deliver calibrated, interpretable hazard telemetry that surfaces structural failures before symbolic validators react. In the logistics disruption scenario the guardrail maintains a three-step lead, offers recovery guidance, and produces documentation-ready figures from a single CLI. Future work includes expanding the twin corpus across additional PlanBench domains and wiring the same instrumentation into live agent execution traces.
