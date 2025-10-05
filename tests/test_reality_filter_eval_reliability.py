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


def test_fever_reliability_gate_emits_admissions() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Transformer reliability gate test")

    detail_path = Path("results/eval/fever_dev/eval_detail.jsonl")
    checkpoint_path = Path("models/reliability_fever_attn_full.pt")
    if not detail_path.exists() or not checkpoint_path.exists():
        pytest.skip("FEVER detail file or reliability checkpoint missing")

    from torch.utils.data import DataLoader, Subset

    from sep_text_manifold.attn_ospace import OspaceTransformer, OspaceTransformerConfig

    from scripts.train_reliability_attn import EvalDetailDataset, collate_batch, run_model

    dataset = EvalDetailDataset(detail_path)

    payload = torch.load(checkpoint_path, map_location="cpu")
    config_blob = payload.get("config")
    if isinstance(config_blob, dict):
        config = OspaceTransformerConfig(**config_blob)
    elif isinstance(config_blob, OspaceTransformerConfig):
        config = config_blob
    else:  # pragma: no cover - defensive guard for incomplete checkpoints
        pytest.skip("Reliability checkpoint missing config payload")

    vocab_payload = payload.get("tokenizer_vocab")
    if isinstance(vocab_payload, dict):
        dataset.tokenizer.vocab = {str(key): int(value) for key, value in vocab_payload.items()}

    subset_size = min(128, len(dataset))
    subset = Subset(dataset, list(range(subset_size)))
    loader = DataLoader(subset, batch_size=32, shuffle=False, collate_fn=collate_batch)

    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):  # pragma: no cover - guard against corrupt checkpoint
        pytest.skip("Reliability checkpoint missing state_dict payload")

    device = torch.device("cuda")
    model = OspaceTransformer(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    tensor_thresholds = eval_mod.default_calibration_thresholds("fever") or (0.2, -0.5)
    admit_threshold, margin_threshold = tensor_thresholds

    admitted_spans = 0
    with torch.inference_mode():
        for batch in loader:
            tensor_batch = {
                key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)
            }
            output = run_model(model, tensor_batch, device, config)
            probabilities = torch.sigmoid(output.admit_logits)
            margins = output.support_margin
            admitted_mask = (probabilities >= admit_threshold) & (margins >= margin_threshold)
            admitted_spans += int(admitted_mask.sum().item())

    assert admitted_spans > 0, "Transformer reliability gate should admit at least one FEVER span"
