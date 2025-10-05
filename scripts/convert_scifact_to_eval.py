#!/usr/bin/env python3
"""Convert SciFact claims and corpus into eval_detail-style records."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

import numpy as np
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from sep_text_manifold.semantic import EmbeddingConfig, SemanticEmbedder

from eval_feature_utils import FeatureExtractor
from scripts.truth_pack_utils import build_truth_pack_from_texts

SUPPORTED_LABEL = "SUPPORTED"
REFUTED_LABEL = "REFUTED"
UNVERIFIABLE_LABEL = "UNVERIFIABLE"


def _normalise(vector: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if vector is None:
        return None
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return None
    return vector / norm


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
        "--neighbor-count",
        type=int,
        default=0,
        help="Number of additional corpus neighbors to attach per claim",
    )
    parser.add_argument(
        "--truth-pack-root",
        type=Path,
        default=Path("analysis/truth_packs"),
        help="Directory where enriched truth packs are written",
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
) -> Tuple[List[EvidenceSentence], Set[Tuple[int, int]]]:
    evidence_sentences: List[EvidenceSentence] = []
    gold_pairs: Set[Tuple[int, int]] = set()
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
                gold_pairs.add((doc_id, sent_idx))
    return evidence_sentences, gold_pairs


def convert_claim(
    claim: ClaimRecord,
    corpus: Dict[int, CorpusEntry],
    extractor: FeatureExtractor,
    split: str,
    *,
    doc_embeddings: Dict[int, np.ndarray],
    neighbor_count: int,
    pack_texts: Dict[str, str],
    doc_text_lookup: Dict[int, str],
) -> Optional[Dict[str, object]]:
    if not claim.claim:
        return None
    question_vec = extractor.vector(claim.claim)
    evidence_items, gold_pairs = gather_evidence(claim, corpus, extractor, question_vec)
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

    # Add full-abstract context sentences for cited documents
    record_docs: Set[int] = set(claim.cited_doc_ids)
    record_docs.update(doc_id for doc_id, _ in gold_pairs)
    for doc_id in sorted(record_docs):
        entry = corpus.get(doc_id)
        if entry is None:
            continue
        for idx, sentence in enumerate(entry.abstract):
            if (doc_id, idx) in gold_pairs:
                continue
            metrics = extractor.metric_vector(sentence, question_vec)
            citation = f"scifact://{doc_id}#{idx}?context=1"
            meta = {"source": "scifact_context", "doc_id": doc_id, "index": idx}
            sentences.append(
                build_sentence(
                    sentence,
                    label,
                    admit=False,
                    metrics=metrics,
                    citation=citation,
                    info=meta,
                )
            )

    q_vec = _normalise(question_vec)
    neighbor_sentences: List[Dict[str, object]] = []
    if neighbor_count > 0 and q_vec is not None:
        scored_docs: List[Tuple[float, int]] = []
        for doc_id, embed in doc_embeddings.items():
            if doc_id in record_docs:
                continue
            score = float(np.dot(q_vec, embed))
            scored_docs.append((score, doc_id))
        if scored_docs:
            scored_docs.sort(key=lambda item: item[0], reverse=True)
            top_docs = [doc_id for _, doc_id in scored_docs[:neighbor_count]]
            for doc_id in top_docs:
                entry = corpus.get(doc_id)
                if entry is None or not entry.abstract:
                    continue
                best_idx = -1
                best_score = float("-inf")
                best_sentence = ""
                for idx, sentence in enumerate(entry.abstract):
                    vec = _normalise(extractor.vector(sentence))
                    if vec is None:
                        continue
                    score = float(np.dot(q_vec, vec))
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                        best_sentence = sentence
                if best_idx < 0:
                    continue
                metrics = extractor.metric_vector(best_sentence, question_vec)
                citation = f"scifact://{doc_id}#{best_idx}?neighbor=1"
                meta = {"source": "scifact_neighbor", "doc_id": doc_id, "index": best_idx}
                neighbor_sentences.append(
                    build_sentence(
                        best_sentence,
                        label,
                        admit=False,
                        metrics=metrics,
                        citation=citation,
                        info=meta,
                    )
                )
                record_docs.add(doc_id)

    sentences.extend(neighbor_sentences)

    # Persist claim text into pack for completeness
    pack_texts.setdefault(f"claim::{claim.id}", claim.claim)
    for doc_id in record_docs:
        text = doc_text_lookup.get(doc_id)
        if text is None:
            continue
        key = f"{doc_id}_{corpus.get(doc_id).title if corpus.get(doc_id) else 'doc'}"
        if key not in pack_texts:
            pack_texts[key] = text

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

    doc_text_lookup: Dict[int, str] = {}
    pack_texts: Dict[str, str] = {}
    for entry in corpus.values():
        text = (entry.title + "\n" + "\n".join(entry.abstract)).strip()
        doc_text_lookup[entry.doc_id] = text
        key = f"{entry.doc_id}_{entry.title or 'doc'}"
        pack_texts[key] = text

    doc_embeddings: Dict[int, np.ndarray] = {}
    for doc_id, text in doc_text_lookup.items():
        vector = extractor.vector(text)
        norm_vec = _normalise(vector)
        if norm_vec is not None:
            doc_embeddings[doc_id] = norm_vec

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w") as out:
        for idx, claim in enumerate(claims, start=1):
            converted = convert_claim(
                claim,
                corpus,
                extractor,
                args.split,
                doc_embeddings=doc_embeddings,
                neighbor_count=args.neighbor_count,
                pack_texts=pack_texts,
                doc_text_lookup=doc_text_lookup,
            )
            if converted is None:
                continue
            out.write(json.dumps(converted) + "\n")
            written += 1
            if args.progress and idx % 200 == 0:
                print(f"processed {idx} claims (written={written})")

    pack_name = f"scifact_{args.split}_full"
    manifest_path = build_truth_pack_from_texts(
        pack_name=pack_name,
        texts=pack_texts,
        output_root=args.truth_pack_root / pack_name,
    )

    print(
        json.dumps(
            {
                "input_claims": str(args.claims),
                "input_corpus": str(args.corpus),
                "output": str(args.output),
                "claims_processed": written,
                "truth_pack_manifest": str(manifest_path),
                "neighbor_count": args.neighbor_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
