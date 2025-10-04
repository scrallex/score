from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import List, Optional

import pytest

import scripts.reality_filter_eval as eval_mod


class DummyReliability:
    def __init__(self, probability: float, margin: float, allow: bool) -> None:
        self._probability = probability
        self._margin = margin
        self._allow = allow

    def score(
        self,
        *,
        question: str,
        candidate: str,
        baseline: str,
        evidence: List[str],
    ) -> tuple[float, float]:
        assert isinstance(question, str)
        assert isinstance(candidate, str)
        assert isinstance(baseline, str)
        assert isinstance(evidence, list)
        return self._probability, self._margin

    def should_admit(self, probability: float, margin: float) -> bool:
        assert pytest.approx(probability) == self._probability
        assert pytest.approx(margin) == self._margin
        return self._allow


class StubEngine:
    def __init__(self, template: eval_mod.SpanEvaluation) -> None:
        self._template = template

    def evaluate_span(self, *args, **kwargs) -> eval_mod.SpanEvaluation:
        return dataclasses.replace(self._template)

    def search_strings(self, token: str, limit: int = 5) -> List[str]:
        return [f"{token}_candidate"]


def make_span_eval(semantic: float) -> eval_mod.SpanEvaluation:
    return eval_mod.SpanEvaluation(
        span="candidate",
        question=None,
        occurrences=5,
        patternability=0.6,
        semantic_similarity=semantic,
        coherence=0.1,
        stability=0.2,
        entropy=0.5,
        rupture=0.0,
        hazard=0.3,
        signature="deadbeef",
        repeat_ok=True,
        hazard_ok=True,
        semantic_ok=False,
        structural_ok=True,
        admitted=False,
        twins=[],
        repair_candidate=None,
    )


def make_context(token: Optional[str] = "foo") -> eval_mod.ClaimContext:
    return eval_mod.ClaimContext(
        claim={"id": "X", "question": "What is foo?", "gold_uris": []},
        baseline_answer="Baseline answer",
        sentences=[],
        snapshots=[],
        token_snapshot=None,
        token=token,
    )


def test_attempt_token_support_with_reliability_allows_admission() -> None:
    thresholds = eval_mod.Thresholds(1, 0.6, 0.3, 0.4, 0.25)
    evaluation = make_span_eval(semantic=0.05)
    engine = StubEngine(evaluation)
    context = make_context()
    outcomes: List[eval_mod.SentenceOutcome] = []
    reliability = DummyReliability(probability=0.9, margin=0.35, allow=True)

    result = eval_mod.attempt_token_support(context, thresholds, engine, outcomes, reliability)
    assert result is not None
    assert result.admit
    assert result.repair_meta and "reliability" in result.repair_meta
    assert pytest.approx(result.repair_meta["reliability"]["probability"]) == 0.9


def test_attempt_token_support_rejects_when_reliability_blocks() -> None:
    thresholds = eval_mod.Thresholds(1, 0.6, 0.3, 0.4, 0.25)
    evaluation = make_span_eval(semantic=0.05)
    engine = StubEngine(evaluation)
    context = make_context()
    outcomes: List[eval_mod.SentenceOutcome] = []
    reliability = DummyReliability(probability=0.1, margin=0.05, allow=False)

    result = eval_mod.attempt_token_support(context, thresholds, engine, outcomes, reliability)
    assert result is None
    assert outcomes == []


def test_reliability_wrapper_requires_attn(monkeypatch, tmp_path):
    monkeypatch.setattr(eval_mod, "attn_torch_available", lambda: False)
    with pytest.raises(RuntimeError):
        eval_mod.ReliabilityModelWrapper(
            tmp_path / "model.pt",
            device="cpu",
            admit_threshold=0.5,
            margin_threshold=0.25,
        )


# TODO: integrate these tests into the gated CI job once the reliability checkpoint
# artefact is published; for now they verify the fallback logic and the
# high-level API contract without requiring a trained model.
