"""Tests for STM adapters covering discriminative feature enrichments."""

from __future__ import annotations

from stm_adapters.pddl_trace import encode_pddl_transition
from stm_adapters.code_trace_adapter import CodeTokenizer


def _structural_tokens(payload: str) -> set[str]:
    return {token for token in payload.split() if token}


def test_pddl_encode_includes_alignment_tokens() -> None:
    state = {"predicates": ["at_a", "holding_crate"]}
    action = {"name": "move", "arguments": ["a", "b"]}
    next_state = {"predicates": ["at_c", "holding_crate"]}

    structural, semantic = encode_pddl_transition(state, action, next_state)

    tokens = _structural_tokens(structural)

    assert "transition__delta_size__few" in tokens
    assert "transition__relative_change__heavy" in tokens
    assert "action__argument_dropout__DRIFT" in tokens
    assert "action__effect_alignment__high" in tokens

    assert "move" in semantic
    assert "arg_unused:b" in semantic


def test_code_tokenizer_python_ast_features() -> None:
    tokenizer = CodeTokenizer()
    patch = "@@\n+def helper(value):\n+    if value > 10:\n+        return value\n+    return value + 1\n"
    step = {
        "action": "apply_patch",
        "metadata": {
            "file_path": "src/example/module.py",
            "patch": patch,
        },
    }

    bundle = tokenizer.encode(step)

    structural = set(bundle.structural)

    assert "edit__py__add_function_signature" in structural
    assert "edit__py__ast_function_def" in structural
    assert any(token.startswith("edit__py__ast_node_count__") for token in structural)
    assert any(token.startswith("edit__delta_additions__") for token in structural)

    assert "func:helper" in bundle.semantic
