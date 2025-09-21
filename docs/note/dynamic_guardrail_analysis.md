# Dynamic Guardrail Sweep (0.01–0.08)

This note summarises the low-guardrail permutation sweeps captured in
`analysis/guardrail_summaries/*_guardrail_sweep.json` and the derived
`docs/note/appendix_guardrail_sensitivity_dynamic.json`. All sweeps use
20k-shuffle permutations against the 100 corrupted traces per domain.
Blocksworld and Mystery were re-run after enriching their twin corpora
with the Logistics gold state (`scripts/enrich_twin_corpus.py`), adding
~7.1k foreground windows and 120 new connector strings per domain.

## Blocksworld

- Mean permutation p-values stay above 0.78 and the minima never fall
  below 0.595, even after tightening the guardrail to 1%. Coverage drops
  to ~1% but the alerts remain statistically indistinguishable from random,
  even after importing additional Logistics twins.
- Lead times plateau around 4–5 steps, so reducing the guardrail further
  does not unlock additional early warning.
- Action: dynamic calibration should defer to the existing 5% guardrail
  until the twin corpus is expanded or additional signal features are
  added; an automated drop would only cut recall without improving
  significance.

## Logistics

- Dropping the guardrail to **2.5%** yields the first statistically
  significant alerts (`p_min ≈ 0.035`) while retaining a 10-step lead and
  ~1.4% weighted coverage.
- The mean p-value remains high (>0.90) until the guardrail reaches 6.5–7%,
  but that expansion collapses the lead to ~3 steps. The low-guardrail
  regime therefore offers better coverage/significance balance than
  pushing the guardrail upward.
- `scripts/calibrate_router.py` now evaluates permutation significance on the
  fly (`--domain-root`) and rewrites the Logistics router to the 2.5% guardrail
  whenever the 5% configuration yields `p_min > 0.05`. The base configuration
  and permutation report are preserved alongside the dynamic artefacts for audit.
- Action: enable dynamic guardrail calibration that ratchets down toward
  2.5% when permutation p-values drift above 0.7 at the 5% baseline. Log
  the reduced coverage so reviewers can see the precision/recall trade.

## Mystery

- P-values remain above 0.08 at every tested guardrail ≤5%; tightening the
  guardrail only reduces coverage to ~2% and shortens the already-small
  leads (<2 steps). The extra Logistics twins provide more candidate
  repairs but do not change permutation significance.
- Action: hold the static guardrail and focus on collecting additional
  mystery-domain traces and twins. Dynamic calibration should not shrink
  the guardrail until a more discriminative signal is available.

## Appendix updates

- `docs/note/appendix_guardrail_sensitivity_dynamic.json` now lists the
  sub-5% guardrail sweep for each domain with the corresponding coverage,
  lead, and permutation statistics. These rows will be surfaced in the
  whitepaper’s sensitivity appendix to show why dynamic calibration is
  only activated for the Logistics domain.
- `make planbench-all` now invokes `scripts/planbench_to_stm.py` with
  `--enrich-from output/planbench_by_domain/logistics/gold_state.json` for
  Blocksworld and Mystery, ensuring the merged twin corpus is reproduced
  on rebuild.
- Additional gold states can be merged by setting
  `PLANBENCH_EXTRA_TWINS="/path/to/extra_state.json ..."` when invoking
  the make target; each state is appended to the enrichment list so sweeps
  can incorporate bug-fix corpora or larger synthetic expansions.
- Recommended whitepaper call-out: “Only the logistics domain benefits
  from a dynamic 2.5% guardrail (p≈0.035, 10-step lead); blocksworld and
  mystery require corpus expansion before low-guardrail alerts become
  significant.”
