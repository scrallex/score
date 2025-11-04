# SEP Applications Suite Overview

1. **STM (Structural Telemetry Manifold)**
   - Converts telemetry/logs into structural bits and surfaces twins + precursors.
   - Adapters: MMS, THEMIS, generic CSV; streaming runtime + `/stm/seen`.
   - Case study: MMS midnight slice (3 twins × 50 windows, 7.4% last-bin density).
   - Docs: `docs/TOOL_QUICKSTART.md`, `docs/note/MMS_Structural_Precursors_v0.1.md`.

2. **Text/Code Manifold**
   - Structural fingerprints for documentation & source code; bridge-string discovery.
   - Use cases: incident runbooks, compliance narratives, code refactor guidance.
   - Outputs: proposals with “why”, theme linking, structural search.

3. **Finance/Signals Manifold**
   - Applies the same mechanics to market/pricing feeds for regime detection.
   - Supports early warning on volatility clusters, cross-asset twins, signal triage.

**Shared Engine**
- Third-state manifold (structural metrics) with percentile guardrail.
- Explainable outputs (bit tokens, signature q-grams, ANN diagnostics).
- Portable CLI/API with reproducible pipelines (`make all`).

**Roadmap**
- Multi-mission telemetry pilots (THEMIS, Cluster, GOES).
- Joint-channel structural bits; automated onset labelling.
- Production streaming for mission ops and grid/IoT fleets.

Use this sheet when positioning the suite to investors or partners.
