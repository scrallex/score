from __future__ import annotations

from dataclasses import replace

import pytest

from reality_filter.engine import SpanEvaluation, TwinResult

import scripts.reality_filter_eval as eval_mod


class _StubEngine:
    """Minimal engine stub for evaluation tests."""

    def search_strings(self, token: str, limit: int = 5) -> list[str]:
        return []


def _make_span(
    *,
    span: str,
    semantic: float,
    occurrences: int = 0,
    patternability: float = 0.0,
    hazard: float = 1.0,
    twins: list[TwinResult] | None = None,
) -> SpanEvaluation:
    return SpanEvaluation(
        span=span,
        question=None,
        occurrences=occurrences,
        patternability=patternability,
        semantic_similarity=semantic,
        coherence=0.0,
        stability=0.0,
        entropy=1.0,
        rupture=0.0,
        hazard=hazard,
        signature="deadbeef",
        repeat_ok=False,
        hazard_ok=False,
        semantic_ok=False,
        structural_ok=False,
        admitted=False,
        twins=twins or [],
        repair_candidate=None,
    )


def test_hallucination_rate_reflects_final_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    thresholds = eval_mod.Thresholds(
        r_min=1,
        lambda_max=0.55,
        sigma_min=0.28,
        structural_threshold=0.2,
        semantic_threshold=0.25,
    )

    base_eval = _make_span(span="original", semantic=0.0)

    def _repair(outcome, _thresholds, _engine):  # pragma: no cover - exercised via test
        repaired_eval = replace(
            base_eval,
            span="repaired",
            semantic_similarity=0.5,
            occurrences=thresholds.r_min,
            patternability=thresholds.structural_threshold,
            hazard=thresholds.lambda_max,
            admitted=True,
            repeat_ok=True,
            hazard_ok=True,
            semantic_ok=True,
            structural_ok=True,
        )
        outcome.sentence = "repaired"
        outcome.evaluation = repaired_eval
        outcome.admit = True
        outcome.repeat_ok = True
        outcome.hazard_ok = True
        outcome.semantic_ok = True
        outcome.structural_ok = True
        outcome.action = "repair"
        return True

    monkeypatch.setattr(eval_mod, "attempt_repair", _repair)

    context = eval_mod.ClaimContext(
        claim={"id": "Q1", "expected": "SUPPORTED"},
        baseline_answer="baseline",
        sentences=["original"],
        snapshots=[base_eval],
        token_snapshot=None,
        token=None,
    )

    result = eval_mod.evaluate_contexts([context], thresholds, _StubEngine(), collect_detail=True)

    assert result.metrics["hallucination_rate"] == pytest.approx(0.0)
    assert result.metrics["repair_yield"] == pytest.approx(1.0)
    detail = result.detail_records[0]
    assert detail["hallucinated"] is False
    assert detail["repaired"] is True


def test_token_support_requires_positive_margin(monkeypatch: pytest.MonkeyPatch) -> None:
    thresholds = eval_mod.Thresholds(
        r_min=1,
        lambda_max=0.55,
        sigma_min=0.28,
        structural_threshold=0.2,
        semantic_threshold=0.25,
    )

    baseline_eval = _make_span(span="baseline", semantic=0.0)
    baseline_outcome = eval_mod.apply_thresholds(baseline_eval, thresholds)
    outcomes = [baseline_outcome]

    gold_uri = "doc://whitepaper#lambda"
    candidate_span = _make_span(
        span="lambda",
        semantic=0.0,
        occurrences=thresholds.r_min,
        patternability=thresholds.structural_threshold,
        hazard=thresholds.lambda_max,
        twins=[
            TwinResult(
                string="lambda",
                occurrences=thresholds.r_min,
                patternability=thresholds.structural_threshold,
                semantic_similarity=0.0,
                hazard=thresholds.lambda_max,
                source=gold_uri,
            )
        ],
    )

    class _TokenEngine:
        def search_strings(self, token: str, limit: int = 5) -> list[str]:
            return ["lambda"] if token == "lambda" else []

        def evaluate_span(self, span: str, **kwargs) -> SpanEvaluation:
            assert span == "lambda"
            return candidate_span

    context = eval_mod.ClaimContext(
        claim={"id": "Q2", "expected": "SUPPORTED", "gold_uris": [gold_uri]},
        baseline_answer="baseline",
        sentences=["baseline"],
        snapshots=[baseline_eval],
        token_snapshot=None,
        token="lambda",
    )

    result = eval_mod.attempt_token_support(context, thresholds, _TokenEngine(), outcomes)

    assert result is None
    assert len(outcomes) == 1
    assert outcomes[0].admit is False
