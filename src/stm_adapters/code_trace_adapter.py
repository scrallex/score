"""Adapter for converting code-agent traces into STM token corpora."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableSequence, Sequence, Tuple

StructuralTokens = List[str]
SemanticTokens = List[str]


@dataclass
class TokenBundle:
    structural: StructuralTokens
    semantic: SemanticTokens


@dataclass
class CodeTokenizer:
    """Tokenises individual agent steps into STM structural/semantic bundles."""

    language_map: Mapping[str, str] | None = None
    rule_aliases: Mapping[str, str] | None = None
    semantic_limit: int = 16

    def encode(self, step: Mapping[str, object]) -> TokenBundle:
        """Convert a single agent step into STM tokens.

        This is a thin orchestrator around feature-specific encoders. The
        implementation is intentionally skeletal; real logic will parse diffs,
        tests, and diagnostics to populate the returned token bundle.
        """
        action = str(step.get("action", ""))
        structural: MutableSequence[str] = []
        semantic: MutableSequence[str] = []

        if action in {"edit", "apply_patch"}:
            bundle = self._encode_edit(step)
            structural.extend(bundle.structural)
            semantic.extend(bundle.semantic)
        if action == "run_tests":
            bundle = self._encode_tests(step)
            structural.extend(bundle.structural)
            semantic.extend(bundle.semantic)
        if action in {"run_lint", "compile"}:
            bundle = self._encode_diagnostics(step)
            structural.extend(bundle.structural)
            semantic.extend(bundle.semantic)
        if action == "search":
            bundle = self._encode_search(step)
            structural.extend(bundle.structural)
            semantic.extend(bundle.semantic)
        if action == "plan_update":
            bundle = self._encode_plan(step)
            structural.extend(bundle.structural)
            semantic.extend(bundle.semantic)

        return TokenBundle(structural=list(dict.fromkeys(structural)), semantic=list(dict.fromkeys(semantic)))

    # Encoder stubs -----------------------------------------------------------------

    def _encode_edit(self, step: Mapping[str, object]) -> TokenBundle:
        file_path = str(step.get("metadata", {}).get("file_path")) if isinstance(step.get("metadata"), Mapping) else ""
        language = self._language_for_path(file_path)
        structural = [f"edit__generic__{language}" if language else "edit__generic"]
        semantic = self._path_tokens(file_path)
        return TokenBundle(structural, semantic[: self.semantic_limit])

    def _encode_tests(self, step: Mapping[str, object]) -> TokenBundle:
        structural = ["test__run"]
        semantic: List[str] = []
        failures = self._metadata_items(step, "failures")
        if failures:
            structural.append("test__fail_any")
            for failure in failures:
                name = str(failure.get("name")) if isinstance(failure, Mapping) else ""
                if name:
                    semantic.append(name)
        return TokenBundle(structural, semantic[: self.semantic_limit])

    def _encode_diagnostics(self, step: Mapping[str, object]) -> TokenBundle:
        structural = ["diagnostic__run"]
        semantic: List[str] = []
        diagnostics = self._metadata_items(step, "diagnostics")
        if diagnostics:
            structural.append("diagnostic__fail_any")
            for diagnostic in diagnostics:
                rule = str(diagnostic.get("rule")) if isinstance(diagnostic, Mapping) else ""
                if rule:
                    rule = self.rule_aliases.get(rule, rule) if self.rule_aliases else rule
                    semantic.append(rule)
        return TokenBundle(structural, semantic[: self.semantic_limit])

    def _encode_search(self, step: Mapping[str, object]) -> TokenBundle:
        results = self._metadata_items(step, "results")
        structural = ["search__hit" if results else "search__no_match"]
        semantic = []
        query = self._metadata_value(step, "query")
        if query:
            semantic.append(str(query))
        return TokenBundle(structural, semantic[: self.semantic_limit])

    def _encode_plan(self, step: Mapping[str, object]) -> TokenBundle:
        summary = self._metadata_value(step, "summary")
        structural = ["plan__update"]
        if summary:
            structural.append("plan__summary_present")
        semantic = []
        next_actions = self._metadata_items(step, "next_actions")
        semantic.extend(str(item) for item in next_actions[: self.semantic_limit])
        return TokenBundle(structural, semantic[: self.semantic_limit])

    # Helpers -----------------------------------------------------------------------

    def _metadata_items(self, step: Mapping[str, object], key: str) -> Sequence[Mapping[str, object]]:
        metadata = step.get("metadata")
        if isinstance(metadata, Mapping):
            value = metadata.get(key)
            if isinstance(value, Sequence):
                return [item for item in value if isinstance(item, Mapping)]
        return []

    def _metadata_value(self, step: Mapping[str, object], key: str) -> object | None:
        metadata = step.get("metadata")
        if isinstance(metadata, Mapping):
            return metadata.get(key)
        return None

    def _language_for_path(self, path: str) -> str:
        suffix = Path(path).suffix if path else ""
        if self.language_map and suffix in self.language_map:
            return self.language_map[suffix]
        if suffix:
            return suffix.lstrip(".")
        return ""

    def _path_tokens(self, path: str) -> List[str]:
        if not path:
            return []
        segments = Path(path).parts
        return [segment for segment in segments if segment]


@dataclass
class CodeTraceAdapter:
    """Adapter turning agent traces into STM token corpora."""

    semantic_suffix: str = "_semantic.txt"
    tokenizer: CodeTokenizer = field(default_factory=CodeTokenizer)

    def run(self, trace_path: Path, output_dir: Path | None = None) -> Path:
        trace_path = Path(trace_path)
        output_dir = output_dir or trace_path.parent / f"{trace_path.stem}_stm"
        output_dir.mkdir(parents=True, exist_ok=True)

        steps = self._load_steps(trace_path)
        structural_lines: List[str] = []
        semantic_lines: List[str] = []
        manifest_sample: List[Dict[str, object]] = []

        for index, step in enumerate(steps):
            bundle = self.tokenizer.encode(step)
            structural_lines.append(" ".join(bundle.structural))
            semantic_lines.append(" ".join(bundle.semantic))
            if index < 5:
                manifest_sample.append({
                    "id": step.get("id"),
                    "action": step.get("action"),
                    "structural_count": len(bundle.structural),
                    "semantic_count": len(bundle.semantic),
                })

        structural_path = output_dir / f"{trace_path.stem}_struct.txt"
        semantic_path = output_dir / f"{trace_path.stem}{self.semantic_suffix}"
        manifest_path = output_dir / f"{trace_path.stem}_manifest.json"

        structural_path.write_text("\n".join(structural_lines), encoding="utf-8")
        semantic_path.write_text("\n".join(semantic_lines), encoding="utf-8")

        manifest = {
            "source": str(trace_path),
            "steps": len(steps),
            "structural": structural_path.name,
            "semantic": semantic_path.name,
            "sample": manifest_sample,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return structural_path

    def _load_steps(self, trace_path: Path) -> Sequence[Mapping[str, object]]:
        text = trace_path.read_text(encoding="utf-8").strip()
        if not text:
            return []
        if "\n" in text and text.lstrip().startswith("{") is False:
            steps: List[Mapping[str, object]] = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if isinstance(data, Mapping):
                    steps.append(data)
            return steps
        data = json.loads(text)
        if isinstance(data, Mapping) and "steps" in data and isinstance(data["steps"], Sequence):
            return [item for item in data["steps"] if isinstance(item, Mapping)]
        if isinstance(data, Sequence):
            return [item for item in data if isinstance(item, Mapping)]
        raise ValueError(f"Unsupported trace format: {trace_path}")

