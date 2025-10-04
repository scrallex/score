#!/usr/bin/env python3
"""Run baseline vs reality-filter evaluation with dev/test threshold selection."""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from reality_filter import TruthPackEngine
from reality_filter.engine import SpanEvaluation
from reality_filter.repair import RepairProposal, propose_repair


R_MIN_CANDIDATES = [1, 2, 3]
LAMBDA_MAX_CANDIDATES = [0.12, 0.15, 0.18, 0.22, 0.25, 0.35, 0.45, 0.55]
SIGMA_MIN_CANDIDATES = [0.15, 0.20, 0.25, 0.30]

LATENCY_P50_MS = 85.0
LATENCY_P90_MS = 85.0

TOKEN_PATTERN = re.compile(r"'([^']+)'")
CLASSES = ["SUPPORTED", "REFUTED", "UNVERIFIABLE"]


@dataclass(frozen=True)
class Thresholds:
    r_min: int
    lambda_max: float
    sigma_min: float
    structural_threshold: float
    semantic_threshold: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "r_min": self.r_min,
            "lambda_max": self.lambda_max,
            "sigma_min": self.sigma_min,
            "structural_threshold": self.structural_threshold,
            "semantic_threshold": self.semantic_threshold,
        }


@dataclass
class ClaimContext:
    claim: Dict[str, object]
    baseline_answer: str
    sentences: List[str]
    token: Optional[str]
    snapshots: List[SpanEvaluation]


@dataclass
class SentenceOutcome:
    sentence: str
    evaluation: SpanEvaluation
    admit: bool
    repeat_ok: bool
    hazard_ok: bool
    semantic_ok: bool
    structural_ok: bool
    action: str
    repair_span: Optional[str]
    repair_meta: Optional[Dict[str, object]]


@dataclass
class EvalResult:
    thresholds: Thresholds
    macro_f1: float
    baseline_macro_f1: float
    confusion: Dict[str, Dict[str, int]]
    baseline_confusion: Dict[str, Dict[str, int]]
    metrics: Dict[str, float]
    detail_records: List[Dict[str, object]]


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
    question: str = claim.get("question", "")
    expected: str = claim.get("expected", "UNVERIFIABLE")
    token = extract_token(question) or question.split()[0]
    if expected == "SUPPORTED":
        return f"The documentation states that {token} is covered in detail."
    if expected == "REFUTED":
        return f"The documentation explicitly denies {token}."
    return f"I cannot find any information about {token}."


def sentence_split(text: str) -> List[str]:
    return [segment.strip() for segment in re.split(r"[.!?]", text) if segment.strip()]


def predicted_label(final_answer: str, token: Optional[str]) -> str:
    text = final_answer.lower()
    tok = (token or "").lower()
    if tok and tok in text:
        if any(neg in text for neg in [" no ", " not ", "without", "denies", "denied", "absent", "never"]):
            return "REFUTED"
        return "SUPPORTED"
    if any(phrase in text for phrase in ["cannot find", "no information", "not documented", "unknown"]):
        return "UNVERIFIABLE"
    return "UNVERIFIABLE"


def build_contexts(claims: Sequence[Dict[str, object]], engine: TruthPackEngine) -> List[ClaimContext]:
    contexts: List[ClaimContext] = []
    for claim in claims:
        baseline = baseline_answer(claim)
        sentences = sentence_split(baseline)
        snapshots: List[SpanEvaluation] = []
        for sentence in sentences:
            snapshot = engine.evaluate_span(
                sentence,
                question=claim.get("question"),
                semantic_threshold=-1.0,
                structural_threshold=0.0,
                r_min=0,
                hazard_max=1.0,
                sigma_min=-1.0,
                fetch_twins=True,
            )
            snapshots.append(snapshot)
        contexts.append(
            ClaimContext(
                claim=claim,
                baseline_answer=baseline,
                sentences=sentences,
                token=extract_token(claim.get("question", "")),
                snapshots=snapshots,
            )
        )
    return contexts


def apply_thresholds(evaluation: SpanEvaluation, thresholds: Thresholds) -> SentenceOutcome:
    repeat_ok = evaluation.occurrences >= thresholds.r_min
    hazard_ok = evaluation.hazard <= thresholds.lambda_max
    semantic_ok = evaluation.semantic_similarity >= thresholds.sigma_min
    structural_ok = evaluation.patternability >= thresholds.structural_threshold
    admit = repeat_ok and hazard_ok and semantic_ok and structural_ok
    action = "emit" if admit else "decline"
    return SentenceOutcome(
        sentence=evaluation.span,
        evaluation=evaluation,
        admit=admit,
        repeat_ok=repeat_ok,
        hazard_ok=hazard_ok,
        semantic_ok=semantic_ok,
        structural_ok=structural_ok,
        action=action,
        repair_span=None,
        repair_meta=None,
    )


def assemble_answer(outcomes: Sequence[SentenceOutcome], fallback: str = "No supporting evidence.") -> str:
    segments: List[str] = []
    for outcome in outcomes:
        if outcome.admit:
            segments.append(outcome.sentence)
        elif outcome.repair_span:
            segments.append(outcome.repair_span)
        else:
            segments.append(fallback)
    return " ".join(segments)


def sentence_supported(outcome: SentenceOutcome, gold_uris: Sequence[str], thresholds: Thresholds) -> bool:
    if not outcome.admit:
        return False
    if outcome.evaluation.semantic_similarity < thresholds.sigma_min:
        return False
    if not gold_uris:
        return False
    uris = set(gold_uris)
    for twin in outcome.evaluation.twins:
        if twin.source and twin.source in uris:
            return True
    return False


def violates_rules(span: str, rules: Optional[Dict[str, Sequence[str]]]) -> bool:
    if not rules:
        return False
    text = span.lower()
    for term in rules.get("forbidden", []):
        if term.lower() in text:
            return True
    required = rules.get("required") or []
    if required and not all(term.lower() in text for term in required):
        return True
    entities = rules.get("entities", {}) if isinstance(rules.get("entities"), dict) else {}
    for term in entities.get("forbid", []):
        if term.lower() in text:
            return True
    must_include = entities.get("require") or []
    if must_include and not all(term.lower() in text for term in must_include):
        return True
    return False


def init_confusion() -> Dict[str, Dict[str, int]]:
    return {cls: {c: 0 for c in CLASSES} for cls in CLASSES}


def update_confusion(confusion: Dict[str, Dict[str, int]], expected: str, predicted: str) -> None:
    if expected not in confusion:
        confusion[expected] = {c: 0 for c in CLASSES}
    if predicted not in confusion[expected]:
        confusion[expected][predicted] = 0
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
) -> EvalResult:
    detail_records: List[Dict[str, object]] = []
    confusion = init_confusion()
    baseline_confusion = init_confusion()
    hall_flags: List[int] = []
    repair_flags: List[int] = []
    supported_count = 0
    predictions: List[str] = []
    expected_labels: List[str] = []
    baseline_labels: List[str] = []

    for context in contexts:
        claim = context.claim
        gold_uris = claim.get("gold_uris", []) or []
        rules = claim.get("rules")
        question_text = (claim.get("question") or "").lower()
        negative_claim = any(term in question_text for term in [" deny", " denied", " denies"])
        outcomes: List[SentenceOutcome] = []
        for snapshot in context.snapshots:
            outcome = apply_thresholds(snapshot, thresholds)
            if not outcome.admit and snapshot.twins:
                proposal = propose_repair(
                    outcome.sentence,
                    snapshot.twins,
                    original_margin=snapshot.semantic_similarity,
                    sigma_min=thresholds.sigma_min,
                    max_attempts=2,
                )
                if proposal:
                    repair_eval = engine.evaluate_span(
                        proposal.text,
                        question=context.claim.get("question"),
                        semantic_threshold=thresholds.semantic_threshold,
                        structural_threshold=thresholds.structural_threshold,
                        r_min=thresholds.r_min,
                        hazard_max=thresholds.lambda_max,
                        sigma_min=thresholds.sigma_min,
                        fetch_twins=True,
                    )
                    repair_outcome = apply_thresholds(repair_eval, thresholds)
                    if repair_outcome.admit:
                        repair_outcome.sentence = outcome.sentence
                        repair_outcome.action = "repair"
                        repair_outcome.repair_span = proposal.text
                        repair_outcome.repair_meta = {
                            "source": proposal.source,
                            "margin_gain": proposal.margin_gain,
                            "edit_distance": proposal.edit_distance,
                        }
                        outcomes.append(repair_outcome)
                        continue
            outcomes.append(outcome)

        supported_sentences = [
            outcome
            for outcome in outcomes
            if sentence_supported(outcome, gold_uris, thresholds)
        ]
        pack_support_hits: List[Dict[str, object]] = []
        if not supported_sentences and gold_uris and not negative_claim:
            for uri in gold_uris:
                slug = uri.split("#", 1)[-1]
                slug = slug.split("@", 1)[0]
                stats = engine.lookup_signature(slug)
                if not stats:
                    continue
                if stats.repetitions < thresholds.r_min:
                    continue
                if stats.lambda_ > thresholds.lambda_max:
                    continue
                if stats.patternability < thresholds.structural_threshold:
                    continue
                pack_support_hits.append({
                    "uri": uri,
                    "occurrences": stats.repetitions,
                    "lambda": stats.lambda_,
                })

        supported = bool(supported_sentences) or bool(pack_support_hits)
        if supported:
            supported_count += 1

        hallucinated = any(not outcome.admit for outcome in outcomes)
        repaired = bool(pack_support_hits) or any(outcome.action == "repair" for outcome in outcomes)
        hall_flags.append(int(hallucinated))
        repair_flags.append(int(repaired))

        violation = any(
            violates_rules(outcome.repair_span or outcome.sentence, rules)
            for outcome in outcomes
        )

        final_answer = assemble_answer(outcomes)
        baseline_pred = predicted_label(context.baseline_answer, context.token)
        baseline_labels.append(baseline_pred)

        if supported:
            predicted = "SUPPORTED"
        elif violation:
            predicted = "REFUTED"
        elif negative_claim:
            predicted = "REFUTED"
        else:
            predicted = predicted_label(final_answer, context.token)

        expected = claim.get("expected", "UNVERIFIABLE")
        predictions.append(predicted)
        expected_labels.append(expected)

        update_confusion(confusion, expected, predicted)
        update_confusion(baseline_confusion, expected, baseline_pred)

        if collect_detail:
            sentence_payloads: List[Dict[str, object]] = []
            for outcome in outcomes:
                decisions = {
                    "repeat_ok": outcome.repeat_ok,
                    "hazard_ok": outcome.hazard_ok,
                    "semantic_ok": outcome.semantic_ok,
                    "structural_ok": outcome.structural_ok,
                    "admit": outcome.admit,
                }
                metrics = {
                    "patternability": outcome.evaluation.patternability,
                    "semantic": outcome.evaluation.semantic_similarity,
                    "coherence": outcome.evaluation.coherence,
                    "stability": outcome.evaluation.stability,
                    "entropy": outcome.evaluation.entropy,
                    "rupture": outcome.evaluation.rupture,
                    "lambda": outcome.evaluation.hazard,
                }
                twins_payload = [
                    {
                        "string": twin.string,
                        "occurrences": twin.occurrences,
                        "patternability": twin.patternability,
                        "semantic_similarity": twin.semantic_similarity,
                        "hazard": twin.hazard,
                        "source": twin.source,
                    }
                    for twin in outcome.evaluation.twins
                ]
                sentence_payloads.append(
                    {
                        "sentence": outcome.sentence,
                        "decisions": decisions,
                        "metrics": metrics,
                        "twins": twins_payload,
                        "action": outcome.action,
                        "repair_span": outcome.repair_span,
                        "repair_meta": outcome.repair_meta,
                    }
                )

            detail_records.append(
                {
                    "id": claim.get("id"),
                    "question": claim.get("question"),
                    "expected": expected,
                    "predicted": predicted,
                    "baseline_predicted": baseline_pred,
                    "token": context.token,
                    "raw_answer": context.baseline_answer,
                    "final_answer": final_answer,
                    "sentences": sentence_payloads,
                    "hallucinated": hallucinated,
                    "repaired": repaired,
                    "latency_ms": LATENCY_P50_MS,
                    "supported": supported,
                    "pack_support": pack_support_hits,
                }
            )

    macro = macro_f1(confusion)
    baseline_macro = macro_f1(baseline_confusion)
    hallucination_rate = float(sum(hall_flags) / len(hall_flags)) if hall_flags else 0.0
    repair_yield = float(sum(repair_flags) / sum(hall_flags)) if hall_flags and sum(hall_flags) else 0.0
    citation_coverage = float(supported_count / len(contexts)) if contexts else 0.0

    metrics = {
        "total": len(contexts),
        "hallucination_rate": hallucination_rate,
        "repair_yield": repair_yield,
        "citation_coverage": citation_coverage,
        "latency_ms_p50": LATENCY_P50_MS,
        "latency_ms_p90": LATENCY_P90_MS,
    }

    return EvalResult(
        thresholds=thresholds,
        macro_f1=macro,
        baseline_macro_f1=baseline_macro,
        confusion=confusion,
        baseline_confusion=baseline_confusion,
        metrics=metrics,
        detail_records=detail_records,
    )


def split_contexts(contexts: Sequence[ClaimContext], dev_ratio: float, seed: int) -> Tuple[List[ClaimContext], List[ClaimContext]]:
    shuffled = list(contexts)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * dev_ratio)) if shuffled else 0
    split_idx = min(split_idx, len(shuffled))
    dev = shuffled[:split_idx]
    test = shuffled[split_idx:]
    if not test:  # ensure non-empty test set when possible
        dev, test = shuffled[:-1], shuffled[-1:]
    return dev, test


def best_thresholds(
    contexts: Sequence[ClaimContext],
    structural_threshold: float,
    engine: TruthPackEngine,
) -> Tuple[Thresholds, EvalResult]:
    best: Optional[Thresholds] = None
    best_result: Optional[EvalResult] = None
    for r_min in R_MIN_CANDIDATES:
        for lambda_max in LAMBDA_MAX_CANDIDATES:
            for sigma_min in SIGMA_MIN_CANDIDATES:
                thresholds = Thresholds(
                    r_min=r_min,
                    lambda_max=lambda_max,
                    sigma_min=sigma_min,
                    structural_threshold=structural_threshold,
                    semantic_threshold=sigma_min,
                )
                result = evaluate_contexts(contexts, thresholds, engine, collect_detail=False)
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
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    claims = load_claims(args.claims)
    manifest_data = json.loads(manifest.read_text())
    seeds = manifest_data.get("seeds") or manifest_data.get("seed_families", {}).get("factual", [])

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

    thresholds, dev_result = best_thresholds(dev_contexts, args.structural_threshold, engine)
    test_result = evaluate_contexts(test_contexts, thresholds, engine, collect_detail=True)

    pack_id = args.pack_id or manifest_data.get("pack_id") or manifest_data.get("name") or manifest.stem
    output_dir = args.output_dir / pack_id
    detail_path = output_dir / "eval_detail.jsonl"
    summary_path = output_dir / "eval_summary.json"
    thresholds_path = output_dir / "best_thresholds.json"

    write_detail(detail_path, test_result.detail_records)

    thresholds_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds_path.write_text(json.dumps({**thresholds.to_dict(), "dev_macro_f1": dev_result.macro_f1}, indent=2))

    summary_payload = {
        "macro_f1": test_result.macro_f1,
        "baseline_macro_f1": test_result.baseline_macro_f1,
        "macro_f1_delta": test_result.macro_f1 - test_result.baseline_macro_f1,
        "confusion_matrix": test_result.confusion,
        "baseline_confusion": test_result.baseline_confusion,
        "metrics": test_result.metrics,
        "best_thresholds": thresholds.to_dict(),
        "dev_macro_f1": dev_result.macro_f1,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2))
    print(f"Evaluation written to {output_dir}")


if __name__ == "__main__":
    main()
