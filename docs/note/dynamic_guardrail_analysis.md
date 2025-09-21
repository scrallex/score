# Dynamic Guardrail Sweep (0.01–0.08)

This note summarises the low-guardrail permutation sweeps captured in
`analysis/guardrail_summaries/*_guardrail_sweep.json` and the derived
`docs/note/appendix_guardrail_sensitivity_dynamic.json`. All sweeps use
20k-shuffle permutations against the 100 corrupted traces per domain.

## Blocksworld

- Mean permutation p-values stay above 0.78 and the minima never fall
  below 0.595, even after tightening the guardrail to 1%. Coverage drops
  to ~1% but the alerts remain statistically indistinguishable from random.
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
- Action: enable dynamic guardrail calibration that ratchets down toward
  2.5% when permutation p-values drift above 0.7 at the 5% baseline. Log
  the reduced coverage so reviewers can see the precision/recall trade.

## Mystery

- P-values remain above 0.08 at every tested guardrail ≤5%; tightening the
  guardrail only reduces coverage to ~2% and shortens the already-small
  leads (<2 steps).
- Action: hold the static guardrail and focus on collecting additional
  mystery-domain traces and twins. Dynamic calibration should not shrink
  the guardrail until a more discriminative signal is available.

## Appendix updates

- `docs/note/appendix_guardrail_sensitivity_dynamic.csv` now lists the
  sub-5% guardrail sweep for each domain with the corresponding coverage,
  lead, and permutation statistics. These rows will be surfaced in the
  whitepaper’s sensitivity appendix to show why dynamic calibration is
  only activated for the Logistics domain.
- Recommended whitepaper call-out: “Only the logistics domain benefits
  from a dynamic 2.5% guardrail (p≈0.035, 10-step lead); blocksworld and
  mystery require corpus expansion before low-guardrail alerts become
  significant.”
