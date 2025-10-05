#!/usr/bin/env python3
"""Convert SciFact claims and corpus into eval_detail-style records."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

import numpy as np
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from sep_text_manifold.semantic import EmbeddingConfig, SemanticEmbedder

from eval_feature_utils import FeatureExtractor

SUPPORTED_LABEL = "SUPPORTED"
REFUTED_LABEL = "REFUTED"
UNVERIFIABLE_LABEL = "UNVERIFIABLE"


@dataclass
class ClaimRecord:
    id: int
    claim: str
    evidence: Dict[str, List[Dict[str, object]]]
    cited_doc_ids: List[int]


@dataclass
class CorpusEntry:
    doc_id: int
    title: str
    abstract: Sequence[str]


@dataclass
class EvidenceSentence:
    text: str
    citation: str
    metrics: Dict[str, float]
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("claims", type=Path, help="Path to claims JSONL (train/dev/test)")
    parser.add_argument("corpus", type=Path, help="Path to SciFact corpus JSONL")
    parser.add_argument("output", type=Path, help="Destination eval_detail.jsonl path")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split name used in generated ids (default: train)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit on number of claims processed",
    )
    parser.add_argument(
        "--semantic-method",
        choices=["auto", "hash", "transformer"],
        default="auto",
        help="Embedding method for semantic similarity (default: auto)",
    )
    parser.add_argument(
        "--semantic-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Transformer model name when using semantic-method auto/transformer",
    )
    parser.add_argument(
        "--semantic-dims",
        type=int,
        default=256,
        help="Hash embedding dimensionality when using semantic-method hash (default: 256)",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print progress every 200 records",
    )
    return parser.parse_args()


def iter_jsonl(path: Path, limit: Optional[int]) -> Iterable[Dict[str, object]]:
    with path.open() as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            stripped = line.strip()
            if not stripped:
                continue
            yield json.loads(stripped)


def load_claims(path: Path, limit: Optional[int]) -> List[ClaimRecord]:
    claims: List[ClaimRecord] = []
    for payload in iter_jsonl(path, limit):
        claims.append(
            ClaimRecord(
                id=int(payload.get("id")),
                claim=str(payload.get("claim") or "").strip(),
                evidence=payload.get("evidence", {}),
                cited_doc_ids=[int(doc_id) for doc_id in payload.get("cited_doc_ids", [])],
            )
        )
    return claims


def load_corpus(path: Path) -> Dict[int, CorpusEntry]:
    corpus: Dict[int, CorpusEntry] = {}
    for payload in iter_jsonl(path, None):
        doc_id = int(payload.get("doc_id"))
        title = str(payload.get("title") or "")
        abstract = payload.get("abstract") or []
        abstract_text = [str(sentence) for sentence in abstract]
        corpus[doc_id] = CorpusEntry(doc_id=doc_id, title=title, abstract=abstract_text)
    return corpus


def classify_label(evidence_items: Sequence[EvidenceSentence]) -> str:
    labels = {item.label.upper() for item in evidence_items if item.label}
    if "SUPPORT" in labels:
        return SUPPORTED_LABEL
    if "CONTRADICT" in labels or "REFUTE" in labels or "REFUTES" in labels:
        return REFUTED_LABEL
    return UNVERIFIABLE_LABEL


def build_sentence(
    text: str,
    label: str,
    *,
    admit: bool,
    metrics: Dict[str, float],
    citation: Optional[str],
    info: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    decisions = {
        "repeat_ok": admit,
        "hazard_ok": label != UNVERIFIABLE_LABEL,
        "semantic_ok": admit,
        "structural_ok": admit,
        "admit": admit,
    }
    payload = {
        "sentence": text,
        "decisions": decisions,
        "metrics": metrics,
        "twins": [],
        "action": "admit" if admit else "decline",
        "repair_span": text if admit else None,
        "repair_meta": info or ( {"source": "scifact"} if admit or info else None ),
        "citations": [citation] if citation else [],
    }
    return payload


def gather_evidence(
    claim: ClaimRecord,
    corpus: Dict[int, CorpusEntry],
    extractor: FeatureExtractor,
    question_vec: Optional[np.ndarray],
) -> List[EvidenceSentence]:
    evidence_sentences: List[EvidenceSentence] = []
    evidence_payload = claim.evidence or {}
    for doc_id_str, rationales in evidence_payload.items():
        try:
            doc_id = int(doc_id_str)
        except (TypeError, ValueError):
            continue
        entry = corpus.get(doc_id)
        sentences = entry.abstract if entry else []
        for rationale in rationales or []:
            label = str(rationale.get("label") or "").upper()
            indices = rationale.get("sentences") or []
            for idx in indices:
                try:
                    sent_idx = int(idx)
                except (TypeError, ValueError):
                    continue
                if 0 <= sent_idx < len(sentences):
                    text = sentences[sent_idx]
                else:
                    text = f"[missing sentence {sent_idx} in doc {doc_id}]"
                metrics = extractor.metric_vector(text, question_vec)
                citation = f"scifact://{doc_id}#{sent_idx}"
                evidence_sentences.append(
                    EvidenceSentence(
                        text=text,
                        citation=citation,
                        metrics=metrics,
                        label=label,
                    )
                )
    return evidence_sentences


def convert_claim(
    claim: ClaimRecord,
    corpus: Dict[int, CorpusEntry],
    extractor: FeatureExtractor,
    split: str,
) -> Optional[Dict[str, object]]:
    if not claim.claim:
        return None
    question_vec = extractor.vector(claim.claim)
    evidence_items = gather_evidence(claim, corpus, extractor, question_vec)
    label = classify_label(evidence_items)

    claim_metrics = extractor.metric_vector(claim.claim, question_vec)
    claim_sentence = build_sentence(
        claim.claim,
        label,
        admit=label == SUPPORTED_LABEL,
        metrics=claim_metrics,
        citation=None,
    )

    sentences = [claim_sentence]
    gold_uris: List[str] = []
    for item in evidence_items:
        gold_uris.append(item.citation)
        admit = label == SUPPORTED_LABEL and item.label.upper() == "SUPPORT"
        meta = {"source": "scifact", "evidence_label": item.label}
        sentence_payload = build_sentence(
            item.text,
            label,
            admit=admit,
            metrics=item.metrics,
            citation=item.citation,
            info=meta,
        )
        sentences.append(sentence_payload)

    record_id = f"SCIFACT_{split}_{claim.id}"
    detail = {
        "id": record_id,
        "question": claim.claim,
        "expected": label,
        "predicted": label,
        "baseline_predicted": UNVERIFIABLE_LABEL,
        "token": None,
        "final_answer": claim.claim if label == SUPPORTED_LABEL else "",
        "baseline_answer": "",
        "sentences": sentences,
        "hallucinated": label != SUPPORTED_LABEL,
        "hallucinated_initial": label != SUPPORTED_LABEL,
        "repaired": False,
        "supported": label == SUPPORTED_LABEL,
        "gold_uris": gold_uris,
        "negative_claim": label == REFUTED_LABEL,
    }
    return detail


def main() -> None:
    args = parse_args()
    extractor_embedder: Optional[SemanticEmbedder]
    embedder_config = EmbeddingConfig(
        method=args.semantic_method,
        model_name=args.semantic_model,
        dims=args.semantic_dims,
    )
    try:
        extractor_embedder = SemanticEmbedder(embedder_config)
    except RuntimeError as exc:
        if args.semantic_method == "transformer":
            raise
        print(f"[convert_scifact_to_eval] Falling back to hash embeddings: {exc}", file=sys.stderr)
        fallback_config = EmbeddingConfig(method="hash", dims=args.semantic_dims)
        extractor_embedder = SemanticEmbedder(fallback_config)

    extractor = FeatureExtractor(embedder=extractor_embedder)
    claims = load_claims(args.claims, args.limit)
    corpus = load_corpus(args.corpus)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w") as out:
        for idx, claim in enumerate(claims, start=1):
            converted = convert_claim(claim, corpus, extractor, args.split)
            if converted is None:
                continue
            out.write(json.dumps(converted) + "\n")
            written += 1
            if args.progress and idx % 200 == 0:
                print(f"processed {idx} claims (written={written})")

    print(
        json.dumps(
            {
                "input_claims": str(args.claims),
                "input_corpus": str(args.corpus),
                "output": str(args.output),
                "claims_processed": written,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
