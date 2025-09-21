# STM Coding Demo

This demo package ships three curated engineering scenarios derived from real agent transcripts. Each scenario includes:

* `baseline.jsonl` — raw agent telemetry without STM guidance.
* `stm.jsonl` — the same task completed with STM dilution/foreground checks and twin repairs.
* `metadata.json` — summary statistics and narrative context.

Use the helper scripts to compare baseline vs. STM runs and regenerate STM-ready corpora via the `CodeTraceAdapter`.

The adapter now extracts syntax-aware features for Python edits. Diff chunks are
classified by their change profile (additions vs. deletions, delta size) and
parsed into AST categories so the guardrail can reason about new function
definitions, control-flow additions, imports, and other semantic cues when
ranking foreground windows.

## Prerequisites

* Python 3.10+
* Install project dependencies (`pip install -e .[test]` if not already). The comparison script only requires the core package.
* Optional: export `STM_BASE_URL` to point at a live STM coprocessor when replaying runs with `ReferenceAgentLoop`.

## Quickstart

```bash
# Generate structural/semantic corpora and summarise metrics
python demo/coding/run_comparison.py

# Regenerate artefacts for a single task + variant
python demo/coding/run_comparison.py --task fix_flaky_test --variant stm

# Replay a trace through the reference agent loop (requires STM_BASE_URL)
python demo/coding/run_replay.py --task rename_service_endpoint --variant stm
```

Reports are written under `demo/coding/output/<task>/<variant>/` mirroring the adapter manifest convention.

## Tasks Overview

| Task | Summary | Baseline Outcome | STM Outcome |
| --- | --- | --- | --- |
| `fix_flaky_test` | Stabilise retry logic by tightening clock tolerance. | 6 steps, 2 failing test loops. | 4 steps, twin patch adopted on first alert. |
| `rename_service_endpoint` | Coordinated rename across handler + tests. | 7 steps, 3 broken imports. | 5 steps, STM surfaces prior rename twin. |
| `resolve_missing_import` | Introduce missing dependency in build. | 5 steps, repeated lint failures. | 3 steps, STM proposes import fix. |

Each replay can be fed into `/stm/enrich` or `/stm/seen` directly to validate dilution, coverage, and twin proposals.
