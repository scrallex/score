# CodeTraceAdapter Specification

## Purpose
The `CodeTraceAdapter` converts raw code agent telemetry into STM-ready structural and semantic streams. It ingests a sequence of tool calls—edits, command runs, and analysis artefacts—and emits rollup files that align with STM ingestion (`/stm/enrich`, `/stm/seen`, `/stm/propose`). The adapter standardises event metadata, extracts structural signals from diffs and logs, and preserves semantic anchors for dilution analysis and twin retrieval.

## Input Contract
Each agent run is serialised as JSON (or yielded incrementally) containing ordered `steps`. Every step records the user-facing action, its artefacts, and timing metadata. The adapter consumes either:

* a JSONL file with one step per line
* a JSON object with `{"steps": [...]}`

Minimum schema per step:

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `id` | string | ✓ | Unique within run; hyphenated sequence preferred (`step-001`). |
| `timestamp` | ISO 8601 string | ✓ | UTC timestamp when artefact became available. |
| `action` | string | ✓ | Canonical verb (see table below). |
| `metadata` | object | ✓ | Action-specific payload (diffs, stdout, tests). |
| `duration_ms` | number | ✗ | Optional latency for weighting. |
| `plan_context` | object | ✗ | Optional plan node info (`goal`, `file`, etc.). |

Canonical `action` values and expected `metadata` shape:

| Action | Metadata keys | Description |
| --- | --- | --- |
| `edit` | `file_path`, `diff` | Unified diff (UTF-8); optional `language`. |
| `apply_patch` | `file_path`, `patch` | Git-style patch; treat identical to `edit`. |
| `run_tests` | `command`, `stdout`, `stderr`, `failures[]` | `failures` items (`name`, `file`, `trace`). |
| `run_lint` | `command`, `stdout`, `stderr`, `diagnostics[]` | Diagnostics as (`rule`, `message`, `location`). |
| `compile` | `command`, `stdout`, `stderr`, `exit_code` | Deduplicate with lint via severity heuristics. |
| `search` | `query`, `results[]` | Results contain (`file_path`, `line`, `match`). |
| `plan_update` | `summary`, `next_actions[]` | Optional, used for plan-drift detection. |

Steps may include arbitrary keys; the adapter ignores unknown properties after logging.

## Token Schema
Tokens are separated into `structural` (scored by STM) and `semantic` (dilution context, twin seeds). They are derived from curated dictionaries plus templated fallbacks.

### Structural Tokens

| Source | Pattern | Example |
| --- | --- | --- |
| File diffs | `edit__<change_type>__<language>` | `edit__function_signature__python` |
| File diffs | `symbol__<operation>__<symbol_kind>` | `symbol__rename__function` |
| File diffs | `dependency__<operation>` | `dependency__add_import` |
| Diff metrics | `diff__lines_<bucket>` | `diff__lines_21_50` |
| Tests | `test__fail__<suite>` | `test__fail__unit_math` |
| Tests | `test__error__<diagnostic>` | `test__error__assertion` |
| Diagnostics | `lint__fail__<rule>` | `lint__fail__flake8_E302` |
| Compiler | `build__fail__<category>` | `build__fail__type_error` |
| Search results | `search__no_match` / `search__hit` | `search__hit` |
| Plan updates | `plan__branch__<kind>` | `plan__branch__repair` |

Token derivation is deterministic; unknown tokens fall back to snake-cased strings with punctuation removed and truncated to 64 characters.

### Semantic Tokens

Semantic tokens anchor the structural signatures and improve dilution separation:

* Full file path segments (`src`, `app`, `utils.py`).
* Symbol names (`resolve_import`, `UserStore`).
* Test identifiers (`tests/test_auth.py::test_login_flow`).
* Diagnostic messages truncated to 6 tokens.
* Query strings for searches.

Semantic tokens are stored alongside structural lines in a `_semantic.txt` file, mirroring the PDDL adapter convention.

## Implementation Sketch

```python
@dataclass
class CodeTraceAdapter:
    semantic_suffix: str = "_semantic.txt"
    tokenizer: CodeTokenizer = field(default_factory=CodeTokenizer)

    def run(self, trace_path: Path, output_dir: Path | None = None) -> Path:
        """Convert an agent trace into STM structural + semantic streams."""
        steps = load_steps(trace_path)
        output_dir = output_dir or trace_path.parent / f"{trace_path.stem}_stm"
        output_dir.mkdir(parents=True, exist_ok=True)

        structural_lines = []
        semantic_lines = []
        manifest = []

        for index, step in enumerate(steps):
            struct_tokens, semantic_tokens = self.tokenizer.encode(step)
            structural_lines.append(" ".join(struct_tokens))
            semantic_lines.append(" ".join(semantic_tokens))
            manifest.append({
                "id": step["id"],
                "action": step["action"],
                "timestamp": step.get("timestamp"),
                "structural_count": len(struct_tokens),
                "semantic_count": len(semantic_tokens),
            })

        structural_path = output_dir / f"{trace_path.stem}_struct.txt"
        semantic_path = output_dir / f"{trace_path.stem}{self.semantic_suffix}"
        structural_path.write_text("\n".join(structural_lines), encoding="utf-8")
        semantic_path.write_text("\n".join(semantic_lines), encoding="utf-8")

        manifest_path = output_dir / f"{trace_path.stem}_manifest.json"
        manifest_payload = {
            "source": str(trace_path),
            "steps": len(steps),
            "structural": structural_path.name,
            "semantic": semantic_path.name,
            "sample": manifest[:5],
        }
        manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
        return structural_path
```

### Tokenizer Responsibilities

`CodeTokenizer.encode(step)` orchestrates feature-specific encoders:

1. **Diff encoder** (`encode_edit`): parses unified diffs, classifies hunks (signature change, branch addition, rename, import). Uses `tree_sitter` if available to disambiguate symbol kinds; falls back to regex heuristics. Emits structural tokens and symbol-level semantic tokens.
2. **Test encoder** (`encode_tests`): tokenises failures, mapping known frameworks (pytest, jest, go test) into canonical categories (assertion, timeout, flaky). Captures failing test IDs.
3. **Diagnostic encoder** (`encode_diagnostics`): maps lint/compile diagnostics into severity buckets and rule IDs; canonicalises by `rule[:1]` to reduce sparsity.
4. **Search encoder** (`encode_search`): indicates whether a search query found matches and the file categories involved.
5. **Plan encoder** (`encode_plan`): converts plan updates into structural drift signals (`plan__branch__repair`, `plan__replan`).

Each encoder returns `(structural_tokens, semantic_tokens)` and the tokenizer merges them, deduplicating via ordered sets to stabilise counts.

### Configuration Hooks

* `language_map`: optional override mapping file suffix (`.py`) → language token (`python`).
* `rule_aliases`: dictionary collapsing lint rule synonyms.
* `semantic_limits`: configurable cap per encoder (default 16 tokens per source).

## Output Artefacts

Given `logs/agent_run.jsonl`, the adapter writes:

```
logs/agent_run_stm/
  agent_run_struct.txt      # structural token windows
  agent_run_semantic.txt    # semantic companion
  agent_run_manifest.json   # encoder summary, counts
```

These artefacts can be POSTed directly via existing CLI wrappers (`stm_cli.enrich`) or mounted into the Docker coprocessor container.

## Next Steps

1. Implement `CodeTokenizer` and encoder modules in `src/stm_adapters/code_trace_adapter.py`.
2. Add unit fixtures covering edit/test/lint encodings.
3. Wire adapter into demo scripts (see `demo/coding`).

