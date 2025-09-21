"""Adapter for converting code-agent traces into STM token corpora."""

from __future__ import annotations

import ast
import json
import re
import textwrap
from collections import Counter
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

        patch, content = self._extract_edit_payload(step)
        diff_tokens, diff_semantic = self._diff_features(patch, language)
        structural.extend(diff_tokens)
        semantic.extend(diff_semantic)

        if language in {"py", "python"}:
            py_tokens, py_semantic = self._python_features(patch, content)
            structural.extend(py_tokens)
            semantic.extend(py_semantic)

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

    def _extract_edit_payload(self, step: Mapping[str, object]) -> Tuple[str, str | None]:
        metadata = step.get("metadata")
        if not isinstance(metadata, Mapping):
            return "", None
        patch = metadata.get("patch") or metadata.get("diff") or ""
        content = metadata.get("content") or metadata.get("new_content")
        return str(patch), str(content) if content is not None else None

    def _diff_features(self, patch: str, language: str) -> Tuple[List[str], List[str]]:
        if not patch:
            return [], []

        additions: List[str] = []
        deletions: List[str] = []
        for line in patch.splitlines():
            if line.startswith("+++") or line.startswith("---"):
                continue
            if line.startswith("+"):
                additions.append(line[1:])
            elif line.startswith("-"):
                deletions.append(line[1:])

        tokens: List[str] = []
        semantics: List[str] = []

        def bucket(count: int) -> str:
            if count <= 0:
                return "none"
            if count <= 3:
                return "small"
            if count <= 8:
                return "medium"
            return "large"

        add_count = len(additions)
        del_count = len(deletions)
        tokens.append(f"edit__delta_additions__{bucket(add_count)}")
        tokens.append(f"edit__delta_deletions__{bucket(del_count)}")

        net = add_count - del_count
        if net == 0:
            net_bucket = "balanced"
        elif net > 0:
            net_bucket = "positive" if net <= 5 else "large_positive"
        else:
            net_bucket = "negative" if net >= -5 else "large_negative"
        tokens.append(f"edit__delta_net__{net_bucket}")

        if add_count and not del_count:
            tokens.append("edit__delta_profile__pure_addition")
        elif del_count and not add_count:
            tokens.append("edit__delta_profile__pure_deletion")
        elif add_count and del_count:
            tokens.append("edit__delta_profile__mixed")

        if language:
            semantics.append(f"delta:{language}:{add_count}:{del_count}")

        return tokens, semantics

    def _python_features(self, patch: str, content: str | None) -> Tuple[List[str], List[str]]:
        additions: List[str] = []
        deletions: List[str] = []
        if patch:
            for line in patch.splitlines():
                if line.startswith("+++") or line.startswith("---"):
                    continue
                if line.startswith("+"):
                    additions.append(line[1:])
                elif line.startswith("-"):
                    deletions.append(line[1:])

        tokens: List[str] = []
        semantics: List[str] = []

        if additions:
            if any(self._looks_like_signature(line, "def") for line in additions):
                tokens.append("edit__py__add_function_signature")
            if any(self._looks_like_signature(line, "class") for line in additions):
                tokens.append("edit__py__add_class_signature")
            if any("import " in line for line in additions):
                tokens.append("edit__py__add_import")
        if deletions:
            if any(self._looks_like_signature(line, "def") for line in deletions):
                tokens.append("edit__py__remove_function_signature")
            if any(self._looks_like_signature(line, "class") for line in deletions):
                tokens.append("edit__py__remove_class_signature")

        ast_tokens, ast_semantics = self._python_ast_tokens(additions, content)
        tokens.extend(ast_tokens)
        semantics.extend(ast_semantics)
        return tokens, semantics

    def _python_ast_tokens(self, additions: Sequence[str], content: str | None) -> Tuple[List[str], List[str]]:
        sources: List[str] = []
        snippet = "\n".join(additions).strip()
        if snippet:
            sources.append(textwrap.dedent(snippet))
            indented = "\n".join(f"    {line}" if line.strip() else "" for line in snippet.splitlines())
            sources.append(f"def _stm_patch_stub():\n{indented}\n")
        if content:
            sources.insert(0, str(content))

        best_tree: ast.AST | None = None
        for source in sources:
            if not source.strip():
                continue
            try:
                best_tree = ast.parse(source)
                break
            except SyntaxError:
                continue

        if best_tree is None:
            if additions:
                return ["edit__py__ast_invalid"], []
            return [], []

        tokens: List[str] = []
        semantics: List[str] = []

        category_counts: Counter[str] = Counter()
        node_total = 0
        max_depth = 0

        def walk(node: ast.AST, depth: int) -> None:
            nonlocal node_total, max_depth
            node_total += 1
            max_depth = max(max_depth, depth)
            category = self._ast_category(node)
            if category:
                category_counts[category] += 1
                if category == "function_def" and isinstance(node, ast.FunctionDef):
                    semantics.append(f"func:{node.name}")
                if category == "class_def" and isinstance(node, ast.ClassDef):
                    semantics.append(f"class:{node.name}")
            for child in ast.iter_child_nodes(node):
                walk(child, depth + 1)

        walk(best_tree, 0)

        tokens.extend(f"edit__py__ast_{key}" for key in sorted(category_counts))

        def bucket_nodes(count: int) -> str:
            if count <= 15:
                return "tiny"
            if count <= 45:
                return "small"
            if count <= 120:
                return "medium"
            return "large"

        def bucket_depth(depth: int) -> str:
            if depth <= 2:
                return "shallow"
            if depth <= 4:
                return "moderate"
            if depth <= 6:
                return "deep"
            return "very_deep"

        tokens.append(f"edit__py__ast_node_count__{bucket_nodes(node_total)}")
        tokens.append(f"edit__py__ast_depth__{bucket_depth(max_depth)}")

        return tokens, semantics

    def _looks_like_signature(self, line: str, keyword: str) -> bool:
        pattern = rf"^{keyword}\s+([A-Za-z_][A-Za-z0-9_]*)"
        return bool(re.match(pattern, line.strip()))

    def _ast_category(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.FunctionDef):
            return "function_def"
        if isinstance(node, ast.AsyncFunctionDef):
            return "async_function_def"
        if isinstance(node, ast.ClassDef):
            return "class_def"
        if isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
            return "loop"
        if isinstance(node, ast.If):
            return "branch"
        if isinstance(node, ast.Try):
            return "try"
        if isinstance(node, ast.With):
            return "context"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "import"
        if isinstance(node, ast.Call):
            return "call"
        if isinstance(node, ast.Return):
            return "return"
        if isinstance(node, ast.Assert):
            return "assert"
        if isinstance(node, ast.Raise):
            return "raise"
        return None


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
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) > 1:
            try:
                parsed_lines = [json.loads(line) for line in lines]
                if all(isinstance(item, Mapping) for item in parsed_lines):
                    return parsed_lines
            except json.JSONDecodeError:
                pass
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            parsed_lines = [json.loads(line) for line in lines]
            return [item for item in parsed_lines if isinstance(item, Mapping)]
        if isinstance(data, Mapping) and "steps" in data and isinstance(data["steps"], Sequence):
            return [item for item in data["steps"] if isinstance(item, Mapping)]
        if isinstance(data, Sequence):
            return [item for item in data if isinstance(item, Mapping)]
        raise ValueError(f"Unsupported trace format: {trace_path}")
