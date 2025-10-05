#!/usr/bin/env python3
"""Run baseline vs reality-filter evaluation with dev/test threshold selection."""

from __future__ import annotations

import argparse
import json
import random
import re
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:  # Optional transformer reliability model
    import torch
except ImportError:  # pragma: no cover - torch optional
    torch = None  # type: ignore[assignment]

try:  # sep_text_manifold attn module is optional
    from sep_text_manifold.attn_ospace import (
        OspaceTransformer,
        OspaceTransformerConfig,
        torch_available as attn_torch_available,
    )
except ImportError:  # pragma: no cover - attn extras may be unavailable
    OspaceTransformer = None  # type: ignore[assignment]
    OspaceTransformerConfig = None  # type: ignore[assignment]

    def attn_torch_available() -> bool:  # type: ignore[override]
        return False

from reality_filter import TruthPackEngine
from reality_filter.engine import SpanEvaluation
from reality_filter.repair import propose_repair

R_MIN_GRID = [1, 2, 3]
LAMBDA_GRID = [0.12, 0.15, 0.18, 0.22, 0.25, 0.35, 0.45, 0.55]
SIGMA_GRID = [0.15, 0.20, 0.25, 0.30]

TOKEN_SUPPORT_MIN_MARGIN = 0.05

TOKEN_PATTERN = re.compile(r"'([^']+)'")
CLASSES = ["SUPPORTED", "REFUTED", "UNVERIFIABLE"]
METRIC_KEYS = (
    "patternability",
    "semantic",
    "coherence",
    "stability",
    "entropy",
    "rupture",
    "lambda",
)

CALIBRATION_PATH = Path("results/analysis/calibration_summary.json")


@dataclass(frozen=True)
class Thresholds:
    r_min: int
    lambda_max: float
    sigma_min: float
    structural_threshold: float
    semantic_threshold: float


@dataclass
class SentenceOutcome:
    sentence: str
    evaluation: SpanEvaluation
    citations: List[str]
    admit: bool
    repeat_ok: bool
    hazard_ok: bool
    semantic_ok: bool
    structural_ok: bool
    action: str
    repair_span: Optional[str]
    repair_meta: Optional[Dict[str, object]]
    reliability: Optional[Dict[str, object]] = None


@dataclass
class ClaimContext:
    claim: Dict[str, object]
    baseline_answer: str
    sentences: List[str]
    snapshots: List[SpanEvaluation]
    token_snapshot: Optional[SpanEvaluation]
    token: Optional[str]


@dataclass
class EvalResult:
    thresholds: Thresholds
    macro_f1: float
    baseline_macro_f1: float
    confusion: Dict[str, Dict[str, int]]
    baseline_confusion: Dict[str, Dict[str, int]]
    metrics: Dict[str, float]
    detail_records: List[Dict[str, object]]
    sanity_flags: List[Dict[str, object]]


@lru_cache(maxsize=1)
def _load_calibration_summary(path: Path = CALIBRATION_PATH) -> Dict[str, Dict[str, object]]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def default_calibration_thresholds(pack_id: Optional[str]) -> Optional[Tuple[float, float]]:
    summary = _load_calibration_summary()
    if not summary:
        return None
    pack_key = None
    if pack_id:
        lowered = pack_id.lower()
        if "scifact" in lowered:
            pack_key = "scifact_val_curriculum"
        elif "hover" in lowered:
            pack_key = "hover_val_fever_adapt"
        elif "fever" in lowered:
            pack_key = "fever_val_curriculum"
        else:
            pack_key = "fever_val_curriculum"
    else:
        pack_key = "fever_val_curriculum"
    record = summary.get(pack_key)
    if not isinstance(record, dict):
        return None
    calibration = record.get("calibration")
    if not isinstance(calibration, dict):
        return None
    admit = calibration.get("admit_threshold")
    margin = calibration.get("margin_threshold")
    if isinstance(admit, (int, float)) and isinstance(margin, (int, float)):
        return float(admit), float(margin)
    return None


class ReliabilityModelWrapper:
    """Thin inference wrapper around the O-space Transformer reliability head."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        device: str,
        admit_threshold: float,
        margin_threshold: float,
        max_evidence: int = 16,
    ) -> None:
        if not attn_torch_available() or torch is None or OspaceTransformer is None:
            raise RuntimeError("Transformer reliability model requested but torch/attn extras unavailable.")
        ckpt = torch.load(checkpoint, map_location=device)
        config_data = ckpt.get("config")
        if isinstance(config_data, dict):
            config = OspaceTransformerConfig(**config_data)
        elif isinstance(config_data, OspaceTransformerConfig):
            config = config_data
        else:
            raise ValueError("Reliability checkpoint missing 'config'.")
        vocab = ckpt.get("tokenizer_vocab")
        if not isinstance(vocab, dict):
            raise ValueError("Reliability checkpoint missing 'tokenizer_vocab'.")
        state_dict = ckpt.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("Reliability checkpoint missing 'state_dict'.")

        self._device = torch.device(device)
        self._config = config
        self._vocab: Dict[str, int] = {str(k): int(v) for k, v in vocab.items()}
        self._pad_id = self._vocab.get("<pad>", 0)
        self._unk_id = self._vocab.get("<unk>", 1)
        self._model = OspaceTransformer(config)
        incompatible = self._model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(  # noqa: T201 - surfaced once per load for operator awareness
                "[reality_filter_eval] Reliability checkpoint loaded with partial state: "
                f"missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)}"
            )
        self._model.to(self._device)
        self._model.eval()
        self._admit_threshold = admit_threshold
        self._margin_threshold = margin_threshold
        self._max_evidence = max(1, max_evidence)
        self._metric_keys = tuple(METRIC_KEYS)
        feature_dim = getattr(config, "evidence_feature_dim", len(self._metric_keys))
        self._feature_dim = feature_dim if feature_dim > 0 else len(self._metric_keys)

    def encode(self, text: str) -> List[int]:
        tokens = text.lower().split()
        if not tokens:
            return [self._unk_id]
        return [self._vocab.get(token, self._unk_id) for token in tokens]

    def _prepare_evidence(
        self,
        evidence: Sequence[Dict[str, object]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens: List[List[int]] = []
        metrics: List[List[float]] = []

        for item in evidence[: self._max_evidence]:
            text = str(item.get("text") or "").strip()
            encoded = self.encode(text)
            tokens.append(encoded)
            raw_metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
            vector = [float(raw_metrics.get(key, 0.0)) for key in self._metric_keys]
            metrics.append(vector)

        if not tokens:
            tokens = [[self._unk_id]]
            metrics = [[0.0] * self._feature_dim]

        mem_count = len(tokens)
        max_len = max(len(seq) for seq in tokens)
        token_tensor = torch.zeros(1, mem_count, max_len, dtype=torch.long, device=self._device)
        mask_tensor = torch.zeros(1, mem_count, max_len, dtype=torch.long, device=self._device)
        feature_tensor = torch.zeros(1, mem_count, self._feature_dim, dtype=torch.float32, device=self._device)

        for idx, seq in enumerate(tokens):
            length = len(seq)
            token_tensor[0, idx, :length] = torch.tensor(seq, dtype=torch.long, device=self._device)
            mask_tensor[0, idx, :length] = 1
            metric_vec = metrics[idx]
            if len(metric_vec) < self._feature_dim:
                metric_vec = metric_vec + [0.0] * (self._feature_dim - len(metric_vec))
            feature_tensor[0, idx] = torch.tensor(metric_vec[: self._feature_dim], dtype=torch.float32, device=self._device)

        return token_tensor, mask_tensor, feature_tensor

    def score(
        self,
        *,
        question: str,
        candidate: str,
        baseline: str,
        evidence: Sequence[Dict[str, object]],
    ) -> Tuple[float, float]:
        aggregate = " \u25cb ".join(token for token in [question, candidate, baseline] if token)
        encoded = self.encode(aggregate)
        input_ids = torch.tensor([encoded], dtype=torch.long, device=self._device)
        attention_mask = (input_ids != self._pad_id).long()
        evidence_tokens, evidence_token_mask, evidence_features = self._prepare_evidence(evidence)

        with torch.no_grad():
            output = self._model(
                input_ids,
                attention_mask=attention_mask,
                evidence_tokens=evidence_tokens,
                evidence_token_mask=evidence_token_mask,
                evidence_features=evidence_features,
            )
        prob = torch.sigmoid(output.admit_logits)[0].item()
        margin = output.support_margin[0].item()
        return prob, margin

    def should_admit(self, probability: float, margin: float) -> bool:
        return probability >= self._admit_threshold and margin >= self._margin_threshold

def load_claims(path: Path) -> List[Dict[str, object]]:
    claims: List[Dict[str, object]] = []
    with path.open() as fh:
        for line in fh:
            if line.strip():
                claims.append(json.loads(line))
    return claims


def extract_token(question: str) -> Optional[str]:
    match = TOKEN_PATTERN.search(question)
    if match:
        return match.group(1)
    return None


def baseline_answer(claim: Dict[str, object]) -> str:
    question = str(claim.get("question", "") or "").strip()
    expected: str = claim.get("expected", "UNVERIFIABLE")
    token = extract_token(question)
    if token is None:
        parts = question.split()
        token = parts[0] if parts else "the claim"
    if expected == "SUPPORTED":
        return f"The documentation states that {token} is covered in detail."
    if expected == "REFUTED":
        return f"The documentation explicitly denies {token}."
    return f"I cannot find any information about {token}."


def sentence_split(text: str) -> List[str]:
    return [segment.strip() for segment in re.split(r"[.!?]", text) if segment.strip()]


def citations_from_evaluation(evaluation: SpanEvaluation) -> List[str]:
    return [t.source for t in evaluation.twins if t.source]


def build_contexts(claims: Sequence[Dict[str, object]], engine: TruthPackEngine) -> List[ClaimContext]:
    contexts: List[ClaimContext] = []
    for claim in claims:
        question = claim.get("question") or ""
        baseline = baseline_answer(claim)
        sentences = sentence_split(baseline)
        snapshots: List[SpanEvaluation] = []
        for sentence in sentences:
            snapshot = engine.evaluate_span(
                sentence,
                question=question,
                semantic_threshold=-1.0,
                structural_threshold=0.0,
                r_min=0,
                hazard_max=1.0,
                sigma_min=-1.0,
                fetch_twins=True,
            )
            snapshots.append(snapshot)
        token = extract_token(question)
        if token is None:
            # Fall back to the first non-trivial word so token support remains available
            for candidate in question.split():
                word = candidate.strip("\"'.,?!;:")
                if len(word) >= 3:
                    token = word
                    break
        token_snapshot: Optional[SpanEvaluation]
        if token:
            token_snapshot = engine.evaluate_span(
                token,
                question=question,
                semantic_threshold=-1.0,
                structural_threshold=0.0,
                r_min=0,
                hazard_max=1.0,
                sigma_min=-1.0,
                fetch_twins=True,
            )
        else:
            token_snapshot = None
        contexts.append(
            ClaimContext(
                claim=claim,
                baseline_answer=baseline,
                sentences=sentences,
                snapshots=snapshots,
                token_snapshot=token_snapshot,
                token=token,
            )
        )
    return contexts


def apply_thresholds(evaluation: SpanEvaluation, thresholds: Thresholds) -> SentenceOutcome:
    repeat_ok = evaluation.occurrences >= thresholds.r_min
    hazard_ok = evaluation.hazard <= thresholds.lambda_max
    semantic_ok = evaluation.semantic_similarity >= thresholds.sigma_min
    structural_ok = evaluation.patternability >= thresholds.structural_threshold
    admit = repeat_ok and hazard_ok and semantic_ok and structural_ok
    citations = citations_from_evaluation(evaluation)
    return SentenceOutcome(
        sentence=evaluation.span,
        evaluation=evaluation,
        citations=citations,
        admit=admit,
        repeat_ok=repeat_ok,
        hazard_ok=hazard_ok,
        semantic_ok=semantic_ok,
        structural_ok=structural_ok,
        action="emit" if admit else "decline",
        repair_span=None,
        repair_meta=None,
    )


def build_reliability_evidence(outcome: SentenceOutcome) -> List[Dict[str, object]]:
    evidence: List[Dict[str, object]] = []
    seen: Set[str] = set()

    def add(text: Optional[str], metrics: Dict[str, float]) -> None:
        if not text:
            return
        key = text.lower()
        if key in seen:
            return
        seen.add(key)
        evidence.append({"text": text, "metrics": metrics})

    add(outcome.sentence, outcome.evaluation.metrics())
    for twin in outcome.evaluation.twins:
        twin_metrics = {
            "patternability": twin.patternability,
            "semantic": twin.semantic_similarity,
            "lambda": twin.hazard,
        }
        add(twin.string, twin_metrics)
    for citation in outcome.citations:
        add(citation, {})
    return evidence


def attempt_repair(outcome: SentenceOutcome, thresholds: Thresholds, engine: TruthPackEngine) -> bool:
    if outcome.admit:
        return False
    twins = outcome.evaluation.twins
    if not twins:
        return False
    proposal = propose_repair(
        outcome.sentence,
        twins,
        original_margin=outcome.evaluation.semantic_similarity,
        sigma_min=thresholds.sigma_min,
        max_attempts=2,
    )
    if proposal is None:
        return False
    repair_eval = engine.evaluate_span(
        proposal.text,
        question=outcome.evaluation.question,
        semantic_threshold=thresholds.semantic_threshold,
        structural_threshold=thresholds.structural_threshold,
        r_min=thresholds.r_min,
        hazard_max=thresholds.lambda_max,
        sigma_min=thresholds.sigma_min,
        fetch_twins=True,
    )
    repaired = apply_thresholds(repair_eval, thresholds)
    if not repaired.admit:
        return False
    outcome.sentence = repaired.sentence
    outcome.evaluation = repair_eval
    outcome.citations = repaired.citations
    outcome.admit = True
    outcome.repeat_ok = repaired.repeat_ok
    outcome.hazard_ok = repaired.hazard_ok
    outcome.semantic_ok = repaired.semantic_ok
    outcome.structural_ok = repaired.structural_ok
    outcome.action = "repair"
    outcome.repair_span = proposal.text
    outcome.repair_meta = {
        "source": proposal.source,
        "margin_gain": proposal.margin_gain,
        "edit_distance": proposal.edit_distance,
    }
    return True


def attempt_token_support(
    context: ClaimContext,
    thresholds: Thresholds,
    engine: TruthPackEngine,
    outcomes: List[SentenceOutcome],
    reliability: Optional[ReliabilityModelWrapper] = None,
    *,
    allow_heuristic: bool = False,
) -> Optional[SentenceOutcome]:
    if context.token is None:
        return None
    if any(outcome.admit for outcome in outcomes):
        return None
    candidates: List[str] = []
    if context.token_snapshot is not None:
        candidates.append(context.token_snapshot.span)
    candidates.extend(engine.search_strings(context.token, limit=5))
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        support_eval = engine.evaluate_span(
            candidate,
            question=context.claim.get("question"),
            semantic_threshold=thresholds.semantic_threshold,
            structural_threshold=thresholds.structural_threshold,
            r_min=thresholds.r_min,
            hazard_max=thresholds.lambda_max,
            sigma_min=thresholds.sigma_min,
            fetch_twins=True,
        )
        support_outcome = apply_thresholds(support_eval, thresholds)
        reliability_meta: Dict[str, object] = {}
        if not support_outcome.admit:
            if reliability is not None:
                evidence_items = build_reliability_evidence(support_outcome)
                prob, margin_pred = reliability.score(
                    question=context.claim.get("question", ""),
                    candidate=candidate,
                    baseline=context.baseline_answer,
                    evidence=evidence_items,
                )
                decision = "admit" if reliability.should_admit(prob, margin_pred) else "reject"
                reliability_meta = {
                    "probability": prob,
                    "margin": margin_pred,
                    "source": "transformer",
                    "decision": decision,
                }
                support_outcome.reliability = reliability_meta
                if decision == "admit":
                    support_outcome.semantic_ok = True
                    support_outcome.admit = True
                    support_outcome.evaluation = support_outcome.evaluation
                else:
                    continue
            elif allow_heuristic:
                gold_uris = context.claim.get("gold_uris", []) or []
                margin = float(support_eval.semantic_similarity)
                required_margin = max(TOKEN_SUPPORT_MIN_MARGIN, thresholds.semantic_threshold * 0.5)
                if (
                    gold_uris
                    and any(uri in gold_uris for uri in support_outcome.citations)
                    and not support_outcome.semantic_ok
                    and support_outcome.repeat_ok
                    and support_outcome.hazard_ok
                    and support_outcome.structural_ok
                    and margin >= required_margin
                ):
                    support_outcome.semantic_ok = True
                    support_outcome.admit = True
                    reliability_meta = {
                        "probability": None,
                        "margin": margin,
                        "source": "heuristic",
                        "decision": "admit",
                    }
                    support_outcome.reliability = reliability_meta
                else:
                    continue
            else:
                continue
        support_outcome.sentence = candidate
        support_outcome.repair_span = candidate
        support_outcome.action = "repair"
        meta: Dict[str, object] = {
            "strategy": "token_support",
            "candidate": candidate,
        }
        if reliability_meta:
            meta["reliability"] = reliability_meta
        meta.setdefault("margin", float(support_eval.semantic_similarity))
        support_outcome.repair_meta = meta
        outcomes.append(support_outcome)
        return support_outcome
    return None


def apply_reliability_gate(
    outcome: SentenceOutcome,
    *,
    reliability: Optional[ReliabilityModelWrapper],
    question: str,
    baseline_answer: str,
    heuristic_admit: bool,
) -> None:
    if reliability is None:
        return
    candidate_text = outcome.repair_span or outcome.sentence
    evidence_items = build_reliability_evidence(outcome)
    prob, margin_pred = reliability.score(
        question=question,
        candidate=candidate_text,
        baseline=baseline_answer,
        evidence=evidence_items,
    )
    decision = "admit" if reliability.should_admit(prob, margin_pred) else "reject"
    outcome.reliability = {
        "probability": prob,
        "margin": margin_pred,
        "source": "transformer",
        "decision": decision,
        "heuristic_admit": heuristic_admit,
    }
    if decision == "admit":
        outcome.admit = True
        outcome.semantic_ok = True
    else:
        outcome.admit = False


def supported_outcomes(
    outcomes: Sequence[SentenceOutcome],
    gold_uris: Sequence[str],
    thresholds: Thresholds,
) -> List[SentenceOutcome]:
    if not gold_uris:
        return []
    uri_set = set(gold_uris)
    supported: List[SentenceOutcome] = []
    for outcome in outcomes:
        if not outcome.admit:
            continue
        margin_ok = outcome.evaluation.semantic_similarity >= thresholds.sigma_min
        token_support = bool(outcome.repair_meta and outcome.repair_meta.get("strategy") == "token_support")
        reliability_admit = bool(
            outcome.reliability and outcome.reliability.get("source") == "transformer" and outcome.reliability.get("decision") == "admit"
        )
        if not margin_ok and not token_support and not reliability_admit:
            continue
        if any(uri in uri_set for uri in outcome.citations):
            supported.append(outcome)
    return supported


def violates_rules(outcomes: Sequence[SentenceOutcome], rules: Optional[Dict[str, object]]) -> bool:
    if not rules:
        return False
    admitted_text = " ".join(
        (outcome.repair_span or outcome.sentence).lower()
        for outcome in outcomes
        if outcome.admit
    )
    if not admitted_text:
        return False
    forbidden = rules.get("forbidden", []) if isinstance(rules, dict) else []
    for term in forbidden:
        if isinstance(term, str) and term.lower() in admitted_text:
            return True
    required = rules.get("required") if isinstance(rules, dict) else None
    if isinstance(required, (list, tuple)) and required:
        if not all(isinstance(term, str) and term.lower() in admitted_text for term in required):
            return True
    entities = rules.get("entities") if isinstance(rules, dict) else None
    if isinstance(entities, dict):
        for term in entities.get("forbid", []):
            if isinstance(term, str) and term.lower() in admitted_text:
                return True
        require_entities = entities.get("require")
        if isinstance(require_entities, (list, tuple)) and require_entities:
            if not all(
                isinstance(term, str) and term.lower() in admitted_text for term in require_entities
            ):
                return True
    return False


def outcome_text(outcome: SentenceOutcome) -> str:
    text = outcome.repair_span or outcome.sentence
    if outcome.citations:
        return f"{text} (see {', '.join(outcome.citations[:2])})"
    return text


def assemble_answer(outcomes: Sequence[SentenceOutcome], fallback: str = "No supporting evidence.") -> str:
    admitted_segments = [outcome_text(outcome) for outcome in outcomes if outcome.admit]
    if admitted_segments:
        return " ".join(admitted_segments)
    return fallback


def baseline_predicted_label(final_answer: str, token: Optional[str]) -> str:
    text = final_answer.lower()
    tok = (token or "").lower()
    if tok and tok in text:
        if any(neg in text for neg in [" no ", " not ", "without", "denies", "denied", "absent", "never"]):
            return "REFUTED"
        return "SUPPORTED"
    if any(phrase in text for phrase in ["cannot find", "no information", "not documented", "unknown"]):
        return "UNVERIFIABLE"
    return "UNVERIFIABLE"


def init_confusion() -> Dict[str, Dict[str, int]]:
    return {cls: {c: 0 for c in CLASSES} for cls in CLASSES}


def update_confusion(confusion: Dict[str, Dict[str, int]], expected: str, predicted: str) -> None:
    confusion.setdefault(expected, {c: 0 for c in CLASSES})
    confusion[expected].setdefault(predicted, 0)
    confusion[expected][predicted] += 1


def macro_f1(confusion: Dict[str, Dict[str, int]]) -> float:
    scores: List[float] = []
    for cls in CLASSES:
        tp = confusion.get(cls, {}).get(cls, 0)
        fp = sum(confusion.get(other, {}).get(cls, 0) for other in CLASSES if other != cls)
        fn = sum(confusion.get(cls, {}).get(other, 0) for other in CLASSES if other != cls)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        scores.append(f1)
    return sum(scores) / len(scores) if scores else 0.0


def evaluate_contexts(
    contexts: Sequence[ClaimContext],
    thresholds: Thresholds,
    engine: TruthPackEngine,
    *,
    collect_detail: bool,
    reliability_model: Optional[ReliabilityModelWrapper] = None,
    heuristic_fallback: bool = False,
) -> EvalResult:
    detail_records: List[Dict[str, object]] = []
    sanity_flags: List[Dict[str, object]] = []
    confusion = init_confusion()
    baseline_confusion = init_confusion()
    hall_initial_list: List[int] = []
    hall_final_list: List[int] = []
    repair_list: List[int] = []
    supported_claims = 0

    for context in contexts:
        claim = context.claim
        gold_uris = claim.get("gold_uris", []) or []
        question = claim.get("question") or ""
        question_text = question.lower()
        negative_claim = any(tok in question_text for tok in [" deny", " denied", " denies", "refute", "refuted"])
        token = context.token

        outcomes: List[SentenceOutcome] = []
        baseline_hallucinated = False
        for snapshot in context.snapshots:
            outcome = apply_thresholds(snapshot, thresholds)
            if not outcome.admit:
                baseline_hallucinated = True
                attempt_repair(outcome, thresholds, engine)
            heuristic_admit = outcome.admit
            if reliability_model is not None:
                apply_reliability_gate(
                    outcome,
                    reliability=reliability_model,
                    question=question,
                    baseline_answer=context.baseline_answer,
                    heuristic_admit=heuristic_admit,
                )
            outcomes.append(outcome)

        token_support = attempt_token_support(
            context,
            thresholds,
            engine,
            outcomes,
            reliability_model,
            allow_heuristic=heuristic_fallback,
        )
        support_hits = supported_outcomes(outcomes, gold_uris, thresholds)
        has_support = bool(support_hits)
        if has_support and not negative_claim:
            supported_claims += 1

        violation = violates_rules(outcomes, claim.get("rules"))

        repair_applied = any(outcome.action == "repair" for outcome in outcomes)
        final_admitted = any(outcome.admit for outcome in outcomes)
        final_hallucinated = not final_admitted

        if support_hits:
            predicted = "REFUTED" if negative_claim else "SUPPORTED"
        elif violation:
            predicted = "REFUTED"
        else:
            predicted = "UNVERIFIABLE"
        baseline_pred = baseline_predicted_label(context.baseline_answer, token)
        expected = claim.get("expected", "UNVERIFIABLE")

        update_confusion(confusion, expected, predicted)
        update_confusion(baseline_confusion, expected, baseline_pred)

        final_answer = assemble_answer(outcomes)
        if predicted == "SUPPORTED" and not support_hits:
            sanity_flags.append(
                {
                    "id": claim.get("id"),
                    "reason": "supported_without_evidence",
                }
            )

        hall_initial_list.append(int(baseline_hallucinated))
        hall_final_list.append(int(final_hallucinated))
        repair_list.append(int(baseline_hallucinated and repair_applied))

        if collect_detail:
            sentence_payloads: List[Dict[str, object]] = []
            for outcome in outcomes:
                sentence_payloads.append(
                    {
                        "sentence": outcome.sentence,
                        "decisions": {
                            "repeat_ok": outcome.repeat_ok,
                            "hazard_ok": outcome.hazard_ok,
                            "semantic_ok": outcome.semantic_ok,
                            "structural_ok": outcome.structural_ok,
                            "admit": outcome.admit,
                        },
                        "metrics": {
                            "patternability": outcome.evaluation.patternability,
                            "semantic": outcome.evaluation.semantic_similarity,
                            "coherence": outcome.evaluation.coherence,
                            "stability": outcome.evaluation.stability,
                            "entropy": outcome.evaluation.entropy,
                            "rupture": outcome.evaluation.rupture,
                            "lambda": outcome.evaluation.hazard,
                            "repetitions": outcome.evaluation.occurrences,
                        },
                        "twins": [
                            {
                                "string": twin.string,
                                "occurrences": twin.occurrences,
                                "patternability": twin.patternability,
                                "semantic_similarity": twin.semantic_similarity,
                                "hazard": twin.hazard,
                                "source": twin.source,
                            }
                            for twin in outcome.evaluation.twins
                        ],
                        "action": outcome.action,
                        "repair_span": outcome.repair_span,
                        "repair_meta": outcome.repair_meta,
                        "citations": outcome.citations,
                        "reliability": outcome.reliability,
                    }
                )

            detail = {
                "id": claim.get("id"),
                "question": question,
                "expected": expected,
                "predicted": predicted,
                "baseline_predicted": baseline_pred,
                "token": token,
                "final_answer": final_answer,
                "baseline_answer": context.baseline_answer,
                "sentences": sentence_payloads,
                "hallucinated": final_hallucinated,
                "hallucinated_initial": baseline_hallucinated,
                "repaired": any(outcome.action == "repair" for outcome in outcomes),
                "supported": has_support,
                "gold_uris": gold_uris,
                "negative_claim": negative_claim,
            }
            reliability_trace = [
                {
                    "sentence": outcome.sentence,
                    **outcome.reliability,
                }
                for outcome in outcomes
                if outcome.reliability
            ]
            if reliability_trace:
                detail["reliability_trace"] = reliability_trace
            detail_records.append(detail)

    macro = macro_f1(confusion)
    baseline_macro = macro_f1(baseline_confusion)
    hallucination_rate = (
        float(sum(hall_final_list) / len(hall_final_list)) if hall_final_list else 0.0
    )
    repair_rate_den = sum(hall_initial_list)
    repair_yield = float(sum(repair_list) / repair_rate_den) if repair_rate_den else 0.0
    citation_coverage = float(supported_claims / len(contexts)) if contexts else 0.0

    metrics = {
        "total": len(contexts),
        "hallucination_rate": hallucination_rate,
        "repair_yield": repair_yield,
        "citation_coverage": citation_coverage,
    }

    return EvalResult(
        thresholds=thresholds,
        macro_f1=macro,
        baseline_macro_f1=baseline_macro,
        confusion=confusion,
        baseline_confusion=baseline_confusion,
        metrics=metrics,
        detail_records=detail_records,
        sanity_flags=sanity_flags,
    )


def split_contexts(contexts: Sequence[ClaimContext], dev_ratio: float, seed: int) -> Tuple[List[ClaimContext], List[ClaimContext]]:
    shuffled = list(contexts)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    if not shuffled:
        return [], []
    split_idx = max(1, int(len(shuffled) * dev_ratio))
    split_idx = min(split_idx, len(shuffled) - 1) if len(shuffled) > 1 else len(shuffled)
    dev = shuffled[:split_idx]
    test = shuffled[split_idx:]
    if not test:
        dev, test = shuffled[:-1], shuffled[-1:]
    return dev, test


def best_thresholds(
    contexts: Sequence[ClaimContext],
    structural_threshold: float,
    engine: TruthPackEngine,
    semantic_threshold: float,
    reliability_model: Optional[ReliabilityModelWrapper],
    heuristic_fallback: bool,
) -> Tuple[Thresholds, EvalResult]:
    best: Optional[Thresholds] = None
    best_result: Optional[EvalResult] = None
    for r_min in R_MIN_GRID:
        for lambda_max in LAMBDA_GRID:
            for sigma_min in SIGMA_GRID:
                thresholds = Thresholds(
                    r_min=r_min,
                    lambda_max=lambda_max,
                    sigma_min=sigma_min,
                    structural_threshold=structural_threshold,
                    semantic_threshold=semantic_threshold,
                )
                result = evaluate_contexts(
                    contexts,
                    thresholds,
                    engine,
                    collect_detail=False,
                    reliability_model=reliability_model,
                    heuristic_fallback=heuristic_fallback,
                )
                if best_result is None or result.macro_f1 > best_result.macro_f1:
                    best = thresholds
                    best_result = result
    assert best is not None and best_result is not None
    return best, best_result


def write_detail(path: Path, records: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_sanity_flags(path: Path, flags: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(list(flags), fh, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/eval"))
    parser.add_argument("--pack-id", type=str)
    parser.add_argument("--semantic-threshold", type=float, default=0.25)
    parser.add_argument("--structural-threshold", type=float, default=0.46)
    parser.add_argument("--r-min", type=int, default=2)
    parser.add_argument("--hazard-max", type=float, default=0.55)
    parser.add_argument("--sigma-min", type=float, default=0.28)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--dev-ratio", type=float, default=0.8)
    parser.add_argument(
        "--skip-threshold-sweep",
        action="store_true",
        help="Use the provided thresholds instead of running the exhaustive sweep on the dev split.",
    )
    parser.add_argument(
        "--disable-reliability",
        action="store_true",
        help="Disable the Transformer reliability gate and fall back to heuristic token-support admissions.",
    )
    parser.add_argument(
        "--reliability-model",
        type=Path,
        help="Path to Transformer reliability checkpoint (defaults to models/reliability_fever_attn_full.pt if available)",
    )
    parser.add_argument(
        "--reliability-device",
        type=str,
        default=None,
        help="Device for the reliability model (cpu/cuda). Defaults to the first available CUDA device.",
    )
    parser.add_argument(
        "--reliability-threshold",
        type=float,
        default=None,
        help="Admission probability threshold for the Transformer reliability gate (defaults to calibrated value).",
    )
    parser.add_argument(
        "--reliability-margin",
        type=float,
        default=None,
        help="Support margin threshold for the Transformer reliability gate (defaults to calibrated value).",
    )
    parser.add_argument(
        "--reliability-max-evidence",
        type=int,
        default=16,
        help="Maximum evidence items passed to the reliability model",
    )
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    claims = load_claims(args.claims)
    manifest_data = json.loads(manifest.read_text())
    seeds = manifest_data.get("seeds") or manifest_data.get("seed_families", {}).get("factual", [])

    pack_id = args.pack_id or manifest_data.get("pack_id") or manifest_data.get("name") or manifest.stem

    calibration_defaults = default_calibration_thresholds(pack_id)
    reliability_threshold = args.reliability_threshold
    reliability_margin = args.reliability_margin
    if calibration_defaults is not None:
        admit_default, margin_default = calibration_defaults
        if reliability_threshold is None:
            reliability_threshold = admit_default
        if reliability_margin is None:
            reliability_margin = margin_default
    else:
        if reliability_threshold is None:
            reliability_threshold = 0.5
        if reliability_margin is None:
            reliability_margin = 0.25

    heuristic_fallback = args.disable_reliability

    reliability_model: Optional[ReliabilityModelWrapper]
    reliability_model = None
    checkpoint_path: Optional[Path] = None
    if not args.disable_reliability:
        checkpoint_path = args.reliability_model
        if checkpoint_path is None:
            default_checkpoint = Path("models/reliability_fever_attn_full.pt")
            if default_checkpoint.exists():
                checkpoint_path = default_checkpoint
        if checkpoint_path is None:
            print("[reality_filter_eval] No reliability checkpoint found; running without Transformer gate.")
        else:
            requested_device = args.reliability_device
            auto_device = (
                "cuda"
                if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available()
                else "cpu"
            )
            device = requested_device or auto_device
            if isinstance(device, str) and device.startswith("cuda"):
                if torch is None or not hasattr(torch, "cuda") or not torch.cuda.is_available():
                    print(
                        "[reality_filter_eval] Requested CUDA device unavailable; falling back to CPU for reliability gate."
                    )
                    device = "cpu"
            try:
                reliability_model = ReliabilityModelWrapper(
                    checkpoint_path,
                    device=device,
                    admit_threshold=float(reliability_threshold),
                    margin_threshold=float(reliability_margin),
                    max_evidence=args.reliability_max_evidence,
                )
                print(
                    f"[reality_filter_eval] Loaded reliability model from {checkpoint_path} on {device}"
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                print(
                    f"[reality_filter_eval] Failed to load reliability model ({exc}). Proceeding without Transformer gate."
                )
                reliability_model = None
    else:
        print("[reality_filter_eval] Transformer reliability gate disabled by flag; using heuristic token support.")

    engine = TruthPackEngine.from_manifest(
        manifest,
        seeds=seeds,
        embedding_method="hash",
        hash_dims=256,
        embedding_min_occ=1,
        lru_size=200_000,
    )

    contexts = build_contexts(claims, engine)
    dev_contexts, test_contexts = split_contexts(contexts, args.dev_ratio, args.split_seed)

    if args.skip_threshold_sweep:
        thresholds = Thresholds(
            r_min=args.r_min,
            lambda_max=args.hazard_max,
            sigma_min=args.sigma_min,
            structural_threshold=args.structural_threshold,
            semantic_threshold=args.semantic_threshold,
        )
        dev_result = evaluate_contexts(
            dev_contexts,
            thresholds,
            engine,
            collect_detail=False,
            reliability_model=reliability_model,
            heuristic_fallback=heuristic_fallback,
        )
    else:
        thresholds, dev_result = best_thresholds(
            dev_contexts,
            args.structural_threshold,
            engine,
            args.semantic_threshold,
            reliability_model,
            heuristic_fallback=heuristic_fallback,
        )
    test_result = evaluate_contexts(
        test_contexts,
        thresholds,
        engine,
        collect_detail=True,
        reliability_model=reliability_model,
        heuristic_fallback=heuristic_fallback,
    )

    output_dir = args.output_dir / pack_id
    detail_path = output_dir / "eval_detail.jsonl"
    summary_path = output_dir / "eval_summary.json"
    thresholds_path = output_dir / "best_thresholds.json"
    sanity_path = output_dir / "sanity_flags.json"

    write_detail(detail_path, test_result.detail_records)
    write_sanity_flags(sanity_path, test_result.sanity_flags)

    thresholds_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds_path.write_text(
        json.dumps(
            {
                "r_min": thresholds.r_min,
                "lambda_max": thresholds.lambda_max,
                "sigma_min": thresholds.sigma_min,
                "structural_threshold": thresholds.structural_threshold,
                "semantic_threshold": thresholds.semantic_threshold,
                "dev_macro_f1": dev_result.macro_f1,
            },
            indent=2,
        )
    )

    summary_payload = {
        "macro_f1": test_result.macro_f1,
        "baseline_macro_f1": test_result.baseline_macro_f1,
        "macro_f1_delta": test_result.macro_f1 - test_result.baseline_macro_f1,
        "confusion_matrix": test_result.confusion,
        "baseline_confusion": test_result.baseline_confusion,
        "metrics": test_result.metrics,
        "best_thresholds": {
            "r_min": thresholds.r_min,
            "lambda_max": thresholds.lambda_max,
            "sigma_min": thresholds.sigma_min,
            "structural_threshold": thresholds.structural_threshold,
            "semantic_threshold": thresholds.semantic_threshold,
        },
        "dev_macro_f1": dev_result.macro_f1,
        "sanity_flags_count": len(test_result.sanity_flags),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, indent=2))
    print(f"Evaluation written to {output_dir}")


if __name__ == "__main__":
    main()
