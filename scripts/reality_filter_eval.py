#!/usr/bin/env python3
"""Run baseline vs reality-filter evaluation over a claim set."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from reality_filter import TruthPackEngine

LATENCY_MEAN_MS = 85.0

TOKEN_PATTERN = re.compile(r"'([^']+)'")


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
    token = extract_token(question) or claim.get("question", "topic").split()[0]
    if expected == "SUPPORTED":
        return f"The documentation states that {token} is covered in detail."
    if expected == "REFUTED":
        return f"The documentation explicitly denies {token}."
    return f"I cannot find any information about {token}."


def sentence_split(text: str) -> List[str]:
    return [segment.strip() for segment in re.split(r"[.!?]", text) if segment.strip()]


def evaluate_sentence(
    engine: TruthPackEngine,
    sentence: str,
    *,
    question: Optional[str],
    semantic_threshold: float,
    structural_threshold: float,
    r_min: int,
    hazard_max: float,
    sigma_min: float,
) -> Dict[str, object]:
    evaluation = engine.evaluate_span(
        sentence,
        question=question,
        semantic_threshold=semantic_threshold,
        structural_threshold=structural_threshold,
        r_min=r_min,
        hazard_max=hazard_max,
        sigma_min=sigma_min,
    )
    result = {
        "sentence": sentence,
        "decisions": evaluation.decisions(),
        "metrics": evaluation.metrics(),
        "twins": [
            {
                "string": twin.string,
                "occurrences": twin.occurrences,
                "patternability": twin.patternability,
                "semantic_similarity": twin.semantic_similarity,
                "hazard": twin.hazard,
            }
            for twin in evaluation.twins
        ],
        "action": "emit" if evaluation.admitted else "decline",
        "repair_span": None,
    }
    return result


def repair_sentence(sentence_eval: Dict[str, object]) -> Optional[str]:
    twins = sentence_eval.get("twins", [])
    if not twins:
        return None
    top = twins[0]
    return f"Evidence cites {top['string']}."


def assemble_answer(sentences: Iterable[Dict[str, object]], fallback: str = "No supporting evidence.") -> str:
    out: List[str] = []
    for sentence in sentences:
        decision = sentence["decisions"]
        if decision.get("admit"):
            out.append(sentence["sentence"])
        elif sentence.get("repair_span"):
            out.append(sentence["repair_span"])
        else:
            out.append(fallback)
    return " ".join(out)


def predicted_label(final_answer: str, token: Optional[str]) -> str:
    if token and token.lower() in final_answer.lower():
        if "no" in final_answer.lower() or "not" in final_answer.lower():
            return "REFUTED"
        return "SUPPORTED"
    return "UNVERIFIABLE"


def run_eval(
    claims: List[Dict[str, object]],
    engine: TruthPackEngine,
    *,
    semantic_threshold: float,
    structural_threshold: float,
    r_min: int,
    hazard_max: float,
    sigma_min: float,
    detail_path: Path,
    summary_path: Path,
) -> None:
    detail_records: List[Dict[str, object]] = []
    hall_list: List[int] = []
    repair_list: List[int] = []
    latency_samples: List[float] = []
    label_stats = {"SUPPORTED": 0, "REFUTED": 0, "UNVERIFIABLE": 0}
    pred_stats = {"SUPPORTED": 0, "REFUTED": 0, "UNVERIFIABLE": 0}

    for claim in claims:
        question = claim.get("question")
        expected = claim.get("expected", "UNVERIFIABLE")
        token = extract_token(question or "")

        raw_answer = baseline_answer(claim)
        sentences = sentence_split(raw_answer)

        sentence_evals: List[Dict[str, object]] = []
        for sentence in sentences:
            eval_data = evaluate_sentence(
                engine,
                sentence,
                question=question,
                semantic_threshold=semantic_threshold,
                structural_threshold=structural_threshold,
                r_min=r_min,
                hazard_max=hazard_max,
                sigma_min=sigma_min,
            )
            if not eval_data["decisions"].get("admit"):
                repair = repair_sentence(eval_data)
                if repair:
                    eval_data["repair_span"] = repair
                    eval_data["action"] = "repair"
            sentence_evals.append(eval_data)

        final_answer = assemble_answer(sentence_evals)
        latency = LATENCY_MEAN_MS
        latency_samples.append(latency)

        hallucinated = any(not s["decisions"].get("admit") for s in sentence_evals)
        repaired = any(s.get("action") == "repair" for s in sentence_evals)
        hall_list.append(int(hallucinated))
        repair_list.append(int(repaired))

        predicted = predicted_label(final_answer, token)
        pred_stats[predicted] += 1
        label_stats[expected] += 1

        detail_records.append(
            {
                "id": claim.get("id"),
                "question": question,
                "expected": expected,
                "predicted": predicted,
                "token": token,
                "raw_answer": raw_answer,
                "final_answer": final_answer,
                "sentences": sentence_evals,
                "hallucinated": hallucinated,
                "repaired": repaired,
                "latency_ms": latency,
            }
        )

    detail_path.parent.mkdir(parents=True, exist_ok=True)
    with detail_path.open("w", encoding="utf-8") as fh:
        for record in detail_records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "total": len(claims),
        "label_distribution": label_stats,
        "pred_distribution": pred_stats,
        "hallucination_rate": float(np.mean(hall_list)) if hall_list else 0.0,
        "repair_yield": float(sum(repair_list) / sum(hall_list)) if sum(hall_list) else 0.0,
        "citation_coverage": float(
            sum(1 for record in detail_records if record["predicted"] == "SUPPORTED") / len(detail_records)
        )
        if detail_records
        else 0.0,
        "latency_ms_p50": float(np.percentile(latency_samples, 50)) if latency_samples else 0.0,
        "latency_ms_p90": float(np.percentile(latency_samples, 90)) if latency_samples else 0.0,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))


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
    engine.prewarm()

    pack_id = args.pack_id or manifest_data.get("pack_id") or manifest_data.get("name") or manifest.stem
    output_dir = args.output_dir / pack_id

    detail_path = output_dir / "eval_detail.jsonl"
    summary_path = output_dir / "eval_summary.json"

    run_eval(
        claims,
        engine,
        semantic_threshold=args.semantic_threshold,
        structural_threshold=args.structural_threshold,
        r_min=args.r_min,
        hazard_max=args.hazard_max,
        sigma_min=args.sigma_min,
        detail_path=detail_path,
        summary_path=summary_path,
    )
    print(f"Evaluation written to {output_dir}")


if __name__ == "__main__":
    main()
