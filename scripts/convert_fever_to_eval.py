#!/usr/bin/env python3
"""Convert FEVER JSONL claims into eval_detail-style records."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from sep_text_manifold.semantic import EmbeddingConfig, SemanticEmbedder

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from eval_feature_utils import FeatureExtractor
from scripts.truth_pack_utils import build_truth_pack_from_texts


LABEL_MAP = {
    "SUPPORTS": "SUPPORTED",
    "REFUTES": "REFUTED",
    "NOT ENOUGH INFO": "UNVERIFIABLE",
}




def _normalise(vector: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if vector is None:
        return None
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return None
    return vector / norm



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fever_json", type=Path, help="Path to FEVER split JSONL (train/dev)")
    parser.add_argument("output", type=Path, help="Destination eval_detail.jsonl path")
    parser.add_argument(
        "--wiki-pages",
        type=Path,
        help="Optional path to FEVER wiki-pages directory or zip archive for evidence text",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split name used in generated ids (default: train)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional record limit for debugging",
    )
    parser.add_argument(
        "--include-predicted",
        action="store_true",
        help="Use predicted_evidence when gold evidence is missing",
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
        help="Number of additional lexical neighbors to attach per claim",
    )
    parser.add_argument(
        "--truth-pack-root",
        type=Path,
        default=Path("analysis/truth_packs"),
        help="Directory where the enriched truth pack should be written",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print progress every 1000 records",
    )
    return parser.parse_args()


def normalize_page(page: Optional[str]) -> Optional[str]:
    if not page:
        return None
    return str(page).replace(" ", "_")


def iter_records(path: Path, limit: Optional[int]) -> Iterable[Dict[str, object]]:
    with path.open() as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            stripped = line.strip()
            if not stripped:
                continue
            yield json.loads(stripped)


def collect_required_pages(
    path: Path,
    limit: Optional[int],
    *,
    include_predicted: bool,
) -> Set[str]:
    pages: Set[str] = set()
    for record in iter_records(path, limit):
        append_pages(record.get("evidence"), pages)
        if include_predicted:
            append_pages(record.get("predicted_evidence"), pages)
    return pages


def append_pages(raw: Optional[object], pages: Set[str]) -> None:
    if not isinstance(raw, list):
        return
    for evidence_set in raw:
        if not isinstance(evidence_set, list):
            continue
        for item in evidence_set:
            if not isinstance(item, list) or len(item) < 3:
                continue
            page = normalize_page(item[2])
            if page:
                pages.add(page)


@dataclass
class Evidence:
    text: str
    citation: str
    metrics: Dict[str, float]


def collect_record_pages(record: Dict[str, object], include_predicted: bool) -> Set[str]:
    pages: Set[str] = set()

    def add(raw: Optional[object]) -> None:
        if not isinstance(raw, list):
            return
        for evidence_set in raw:
            if not isinstance(evidence_set, list):
                continue
            for item in evidence_set:
                if not isinstance(item, list) or len(item) < 3:
                    continue
                page = normalize_page(item[2])
                if page:
                    pages.add(page)

    add(record.get("evidence"))
    if include_predicted:
        add(record.get("predicted_evidence"))
    return pages


class WikiEvidenceStore:
    """Load FEVER wiki sentences for the required subset of pages."""

    def __init__(self, root: Path, required_pages: Set[str]) -> None:
        self._sentences: Dict[str, List[str]] = {}
        self._raw_text: Dict[str, str] = {}
        self._load(root, required_pages)

    def _load(self, root: Path, required: Set[str]) -> None:
        if not required:
            return
        sources: List[Path] = []
        if root.is_dir():
            sources.extend(sorted(root.glob("*.jsonl")))
            sources.extend(sorted(root.glob("wiki_*.jsonl")))
        elif root.is_file():
            sources.append(root)
        if not sources:
            raise FileNotFoundError(f"No wiki jsonl files found in {root}")

        remaining = set(required)
        for source in sources:
            if not remaining:
                break
            with source.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    page = normalize_page(payload.get("id"))
                    if not page or page not in remaining:
                        continue
                    text = payload.get("text", "")
                    sentences = parse_wiki_sentences(text)
                    self._sentences[page] = sentences
                    self._raw_text[page] = "\n".join(sentences)
                    remaining.discard(page)

    def get(self, page: Optional[str], sentence_idx: Optional[int]) -> Optional[str]:
        norm = normalize_page(page)
        if norm is None or sentence_idx is None:
            return None
        sentences = self._sentences.get(norm)
        if sentences is None:
            return None
        if 0 <= sentence_idx < len(sentences):
            return sentences[sentence_idx]
        return None

    def all_sentences(self, page: Optional[str]) -> Sequence[str]:
        norm = normalize_page(page)
        if norm is None:
            return []
        return self._sentences.get(norm, [])

    def article_text(self, page: Optional[str]) -> Optional[str]:
        norm = normalize_page(page)
        if norm is None:
            return None
        return self._raw_text.get(norm)


def select_neighbor_sentences(
    *,
    record_pages: Set[str],
    store: Optional[WikiEvidenceStore],
    extractor: FeatureExtractor,
    question_vector: Optional[np.ndarray],
    gold_keys: Set[Tuple[str, int]],
    neighbor_count: int,
) -> List[Tuple[str, str, Dict[str, float]]]:
    if neighbor_count <= 0 or store is None or question_vector is None or not record_pages:
        return []

    q_vec = _normalise(question_vector)
    if q_vec is None:
        return []

    candidates: List[Tuple[float, str, int, str]] = []
    for page in record_pages:
        sentences = store.all_sentences(page)
        if not sentences:
            continue
        for idx, text in enumerate(sentences):
            key = (page, idx)
            if key in gold_keys:
                continue
            vec = _normalise(extractor.vector(text))
            if vec is None:
                continue
            score = float(np.dot(q_vec, vec))
            candidates.append((score, page, idx, text))

    if not candidates:
        return []

    candidates.sort(key=lambda item: item[0], reverse=True)
    top = candidates[:neighbor_count]
    results: List[Tuple[str, str, Dict[str, float]]] = []
    for _, page, idx, text in top:
        citation = f"wiki://{page}#{idx}?neighbor=1"
        metrics = extractor.metric_vector(text, question_vector)
        results.append((text, citation, metrics))
    return results


def parse_wiki_sentences(text: str) -> List[str]:
    sentences: List[Tuple[int, str]] = []
    for raw in text.split("\n"):
        parts = raw.split("\t", 1)
        if len(parts) != 2:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        sentences.append((index, parts[1]))
    sentences.sort(key=lambda item: item[0])
    return [item[1] for item in sentences]




def evidence_from_record(
    record: Dict[str, object],
    store: Optional[WikiEvidenceStore],
    *,
    include_predicted: bool,
    extractor: FeatureExtractor,
    question_vector: Optional[np.ndarray],
) -> Tuple[List[Evidence], Set[Tuple[str, int]]]:
    collected: Dict[Tuple[str, int], Evidence] = {}

    def add_from(raw: Optional[object]) -> None:
        if not isinstance(raw, list):
            return
        for evidence_set in raw:
            if not isinstance(evidence_set, list):
                continue
            for item in evidence_set:
                if not isinstance(item, list) or len(item) < 4:
                    continue
                page = normalize_page(item[2])
                try:
                    sentence_idx = int(item[3])
                except (TypeError, ValueError):
                    continue
                key = (page or "", sentence_idx)
                if key in collected:
                    continue
                text = store.get(page, sentence_idx) if store is not None else None
                if not text:
                    text = f"{page or 'unknown'} (sentence {sentence_idx})"
                citation = f"wiki://{page}#{sentence_idx}" if page else f"wiki://unknown#{sentence_idx}"
                metrics = extractor.metric_vector(text, question_vector)
                collected[key] = Evidence(text=text, citation=citation, metrics=metrics)

    add_from(record.get("evidence"))
    if include_predicted:
        add_from(record.get("predicted_evidence"))

    items = list(collected.items())
    items.sort(key=lambda kv: kv[0])
    evidence = [kv[1] for kv in items]
    return evidence, set(collected.keys())

def build_sentence(
    text: str,
    label: str,
    *,
    admit: bool,
    metrics: Dict[str, float],
    citation: Optional[str],
    meta: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    decisions = {
        "repeat_ok": admit,
        "hazard_ok": label != "UNVERIFIABLE",
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
        "repair_meta": meta if meta is not None else ({"source": "fever"} if admit else None),
        "citations": [citation] if citation else [],
    }
    return payload


def convert_record(
    record: Dict[str, object],
    *,
    store: Optional[WikiEvidenceStore],
    split: str,
    include_predicted: bool,
    extractor: FeatureExtractor,
    neighbor_count: int,
    text_sink: Dict[str, str],
) -> Optional[Dict[str, object]]:
    label_raw = str(record.get("label") or "").upper()
    mapped = LABEL_MAP.get(label_raw)
    if mapped is None:
        return None
    claim = str(record.get("claim") or "").strip()
    if not claim:
        return None

    question_vector = extractor.vector(claim)

    record_pages = collect_record_pages(record, include_predicted)
    if store is not None:
        for page in record_pages:
            if page not in text_sink:
                article = store.article_text(page)
                if article:
                    text_sink[page] = article

    evidence_items, gold_keys = evidence_from_record(
        record,
        store,
        include_predicted=include_predicted,
        extractor=extractor,
        question_vector=question_vector,
    )
    gold_uris = [item.citation for item in evidence_items]
    claim_metrics = extractor.metric_vector(claim, question_vector)
    claim_sentence = build_sentence(
        claim,
        mapped,
        admit=mapped == "SUPPORTED",
        metrics=claim_metrics,
        citation=None,
    )

    evidence_sentences = [
        build_sentence(
            item.text,
            mapped,
            admit=mapped == "SUPPORTED",
            metrics=item.metrics,
            citation=item.citation,
        )
        for item in evidence_items
    ]

    neighbor_payloads = select_neighbor_sentences(
        record_pages=record_pages,
        store=store,
        extractor=extractor,
        question_vector=question_vector,
        gold_keys=gold_keys,
        neighbor_count=neighbor_count,
    )
    neighbor_sentences = [
        build_sentence(
            text,
            mapped,
            admit=False,
            metrics=metrics,
            citation=citation,
            meta={"source": "fever_neighbor"},
        )
        for text, citation, metrics in neighbor_payloads
    ]

    sentences = [claim_sentence] + evidence_sentences + neighbor_sentences
    record_id = record.get("id")
    detail = {
        "id": f"FEVER_{split}_{record_id}",
        "question": claim,
        "expected": mapped,
        "predicted": mapped,
        "baseline_predicted": "UNVERIFIABLE",
        "token": None,
        "final_answer": claim,
        "baseline_answer": "",
        "sentences": sentences,
        "hallucinated": mapped != "SUPPORTED",
        "hallucinated_initial": mapped != "SUPPORTED",
        "repaired": False,
        "supported": mapped == "SUPPORTED",
        "gold_uris": gold_uris,
        "negative_claim": mapped == "REFUTED",
    }
    return detail


def main() -> None:
    args = parse_args()
    if args.wiki_pages is not None and not args.wiki_pages.exists():
        raise FileNotFoundError(f"wiki-pages path not found: {args.wiki_pages}")

    required_pages: Set[str] = set()
    store: Optional[WikiEvidenceStore] = None

    if args.wiki_pages is not None:
        required_pages = collect_required_pages(
            args.fever_json,
            args.limit,
            include_predicted=args.include_predicted,
        )
        store = WikiEvidenceStore(args.wiki_pages, required_pages)

    embedder_config = EmbeddingConfig(
        method=args.semantic_method,
        model_name=args.semantic_model,
        dims=args.semantic_dims,
    )
    try:
        semantic_embedder = SemanticEmbedder(embedder_config)
    except RuntimeError as exc:
        if args.semantic_method == "transformer":
            raise
        print(f"[convert_fever_to_eval] Falling back to hash embeddings: {exc}", file=sys.stderr)
        semantic_embedder = SemanticEmbedder(EmbeddingConfig(method="hash", dims=args.semantic_dims))
    extractor = FeatureExtractor(embedder=semantic_embedder)

    if store is None:
        raise RuntimeError(
            "FEVER conversion requires wiki-pages path to build enriched truth pack"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    missing_label = 0
    page_texts: Dict[str, str] = {}

    with args.output.open("w") as out:
        for idx, record in enumerate(iter_records(args.fever_json, args.limit), start=1):
            converted = convert_record(
                record,
                store=store,
                split=args.split,
                include_predicted=args.include_predicted,
                extractor=extractor,
                neighbor_count=args.neighbor_count,
                text_sink=page_texts,
            )
            if converted is None:
                missing_label += 1
                continue
            record_id = converted.get("id") or idx
            if isinstance(record_id, str):
                page_texts.setdefault(f"claim::{record_id}", converted["question"])
            out.write(json.dumps(converted) + "\n")
            written += 1
            if args.progress and idx % 1000 == 0:
                print(f"processed {idx} records (written={written})")

    if store is not None:
        for page in required_pages:
            if page not in page_texts:
                article = store.article_text(page)
                if article:
                    page_texts[page] = article

    pack_name = f"fever_{args.split}_full"
    manifest_path = build_truth_pack_from_texts(
        pack_name=pack_name,
        texts=page_texts,
        output_root=args.truth_pack_root / pack_name,
    )

    print(
        json.dumps(
            {
                "input": str(args.fever_json),
                "output": str(args.output),
                "records_written": written,
                "skipped": missing_label,
                "required_pages": len(required_pages),
                "truth_pack_manifest": str(manifest_path),
                "neighbor_count": args.neighbor_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
