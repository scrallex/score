# SEP Documentation (Core Set)

The documentation set has been trimmed to the canonical references that describe the live quantum-metrics trading system. Everything else—including historical plans, onboarding material, diagram catalogues, and program logs—now lives under [`docs/_archive/`](./_archive/).

## Core References

1. [`01_System_Concepts.md`](01_System_Concepts.md) — architecture, service topology, and Valkey schema.
2. [`02_Operations_Runbook.md`](02_Operations_Runbook.md) — day-to-day operations, warmup orchestration, and recovery playbooks.
3. [`03_Whitepaper.md`](03_Whitepaper.md) — theoretical background for the quantum metrics stack.
4. [`04_Developer_Guide.md`](04_Developer_Guide.md) — engineering conventions, API contract, and implementation guide.
5. [`05_Core_Trading_Loop.md`](05_Core_Trading_Loop.md) — execution contract for the backend engine (inputs, outputs, guardrails).
6. [`09_Live_Recompute_Pipeline.md`](09_Live_Recompute_Pipeline.md) — rolling evaluator + allocator-lite loop, configuration knobs, telemetry.
7. [`11_Gating_and_Allocator_Tuning.md`](11_Gating_and_Allocator_Tuning.md) — guard thresholds, hysteresis, and allocator scoring guidance.

Archived materials can be referenced as needed but are no longer part of the active documentation surface.
