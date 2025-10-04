from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional, Sequence

from reality_filter.engine import TwinResult


ENTITY_PATTERN = re.compile(r"\b(?:\d+[\w%]*)\b")


@dataclass
class RepairProposal:
    text: str
    source: Optional[str]
    margin: float
    edit_distance: float
    margin_gain: float


def extract_entities(text: str) -> set[str]:
    return {match.group(0) for match in ENTITY_PATTERN.finditer(text)}


def _edit_distance_ratio(original: str, candidate: str) -> float:
    matcher = SequenceMatcher(None, original.lower(), candidate.lower())
    return 1.0 - matcher.ratio()


def propose_repair(
    span: str,
    twins: Sequence[TwinResult],
    *,
    original_margin: float,
    sigma_min: float,
    max_attempts: int = 2,
) -> Optional[RepairProposal]:
    if not twins:
        return None
    original_entities = extract_entities(span)
    attempts = 0
    best_proposal: Optional[RepairProposal] = None
    for twin in twins:
        if attempts >= max_attempts:
            break
        attempts += 1
        candidate = twin.string.strip()
        if not candidate:
            continue
        candidate_entities = extract_entities(candidate)
        if original_entities and not original_entities.issubset(candidate_entities):
            continue
        margin = float(twin.semantic_similarity)
        if margin < sigma_min:
            continue
        edit_distance = _edit_distance_ratio(span, candidate)
        proposal = RepairProposal(
            text=candidate,
            source=twin.source,
            margin=margin,
            edit_distance=edit_distance,
            margin_gain=margin - original_margin,
        )
        best_proposal = proposal
        break
    return best_proposal


__all__ = ["RepairProposal", "extract_entities", "propose_repair"]
