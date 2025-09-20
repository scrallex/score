"""Adapter for converting PDDL traces into STM-friendly tokens."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


def _normalise_predicate(predicate: str) -> str:
    return predicate.strip().lower().replace(" ", "_")


def _coerce_predicates(state: Iterable[str] | dict | None) -> List[str]:
    if state is None:
        return []
    if isinstance(state, dict):
        items = state.get("predicates") or state.get("state") or []
    else:
        items = state
    return [_normalise_predicate(str(item)) for item in items]


def _parse_action(action: str | dict) -> Tuple[str, List[str]]:
    if isinstance(action, dict):
        name = action.get("name", "")
        args = action.get("arguments") or action.get("args") or []
        args = [str(arg) for arg in args]
    else:
        text = str(action).strip()
        if " " in text:
            name, rest = text.split(" ", 1)
            args = rest.replace("(", " ").replace(")", " ").split()
        else:
            name, args = text, []
    return name.lower(), [arg.lower() for arg in args]


def encode_pddl_transition(
    state: Iterable[str] | dict | None,
    action: str | dict,
    next_state: Iterable[str] | dict | None,
) -> Tuple[str, List[str]]:
    """Convert a PDDL transition into structural and semantic tokens.

    Returns
    -------
    tuple
        ``(structural_tokens, semantic_tokens)`` where ``structural_tokens``
        is a single space-delimited string suitable for ingestion and
        ``semantic_tokens`` is a list that can be stored for semantic
        dilution analysis.
    """
    previous = set(_coerce_predicates(state))
    after = set(_coerce_predicates(next_state))
    added = sorted(after - previous)
    removed = sorted(previous - after)
    persisted = sorted(after & previous)

    action_name, action_args = _parse_action(action)

    structural_tokens: List[str] = []
    for predicate in added:
        structural_tokens.append(f"{predicate}__UP")
    for predicate in removed:
        structural_tokens.append(f"{predicate}__RANGEEXP")
    for predicate in persisted:
        structural_tokens.append(f"{predicate}__ZPOS")
    if action_name:
        structural_tokens.append(f"action__{action_name}__ACCEL")
    for idx, arg in enumerate(action_args):
        structural_tokens.append(f"action_arg{idx}__{arg}__ACCEL")
    if added and removed:
        structural_tokens.append("transition__RUPTURE")

    semantic_tokens: List[str] = []
    if action_name:
        semantic_tokens.append(action_name)
    semantic_tokens.extend(added)
    semantic_tokens.extend(f"not_{pred}" for pred in removed)

    return " ".join(structural_tokens), semantic_tokens


@dataclass
class PDDLTraceAdapter:
    """Adapter turning VAL-style JSON traces into STM token corpora."""

    semantic_suffix: str = "_semantic.txt"

    def run(self, trace_path: Path, output_dir: Path | None = None) -> Path:
        trace_path = Path(trace_path)
        output_dir = output_dir or trace_path.parent / f"{trace_path.stem}_stm"
        output_dir.mkdir(parents=True, exist_ok=True)

        data = json.loads(trace_path.read_text(encoding="utf-8"))
        transitions = data.get("transitions") or data.get("trace") or []
        if not isinstance(transitions, Sequence):
            raise ValueError(f"Trace file {trace_path} does not contain a transitions list")

        structural_lines: List[str] = []
        semantic_lines: List[str] = []
        for transition in transitions:
            if isinstance(transition, dict):
                state = transition.get("state")
                next_state = transition.get("next_state") or transition.get("nextState")
                action = transition.get("action")
            else:
                raise ValueError("Transitions must be dictionaries with state/action/next_state")
            structural, semantic = encode_pddl_transition(state, action, next_state)
            structural_lines.append(structural)
            semantic_lines.append(" ".join(semantic))

        structural_path = output_dir / f"{trace_path.stem}_struct.txt"
        semantic_path = output_dir / f"{trace_path.stem}{self.semantic_suffix}"
        structural_path.write_text("\n".join(structural_lines), encoding="utf-8")
        semantic_path.write_text("\n".join(semantic_lines), encoding="utf-8")

        manifest = {
            "source": str(trace_path),
            "structural": structural_path.name,
            "semantic": semantic_path.name,
            "transition_count": len(structural_lines),
        }
        (output_dir / f"{trace_path.stem}_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        return structural_path
