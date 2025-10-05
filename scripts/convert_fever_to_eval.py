#!/usr/bin/env python3
"""Convert FEVER JSONL claims into eval_detail-style records."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from sep_text_manifold.semantic import EmbeddingConfig, SemanticEmbedder
from eval_feature_utils import FeatureExtractor


LABEL_MAP = {
    "SUPPORTS": "SUPPORTED",
    "REFUTES": "REFUTED",
    "NOT ENOUGH INFO": "UNVERIFIABLE",
}




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


class WikiEvidenceStore:
    """Load FEVER wiki sentences for the required subset of pages."""

    def __init__(self, root: Path, required_pages: Set[str]) -> None:
        self._data: Dict[str, List[str]] = {}
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
                    self._data[page] = sentences
                    remaining.discard(page)

    def get(self, page: Optional[str], sentence_idx: Optional[int]) -> Optional[str]:
        norm = normalize_page(page)
        if norm is None or sentence_idx is None:
            return None
        sentences = self._data.get(norm)
        if sentences is None:
            return None
        if 0 <= sentence_idx < len(sentences):
            return sentences[sentence_idx]
        return None


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
) -> List[Evidence]:
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

    return list(collected.values())

def build_sentence(text: str, label: str, *, admit: bool, metrics: Dict[str, float], citation: Optional[str]) -> Dict[str, object]:
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
        "repair_meta": {"source": "fever"} if admit else None,
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
) -> Optional[Dict[str, object]]:
    label_raw = str(record.get("label") or "").upper()
    mapped = LABEL_MAP.get(label_raw)
    if mapped is None:
        return None
    claim = str(record.get("claim") or "").strip()
    if not claim:
        return None

    question_vector = extractor.vector(claim)

    evidence_items = evidence_from_record(
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

    sentences = [claim_sentence] + evidence_sentences
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    missing_label = 0

    with args.output.open("w") as out:
        for idx, record in enumerate(iter_records(args.fever_json, args.limit), start=1):
            converted = convert_record(
                record,
                store=store,
                split=args.split,
                include_predicted=args.include_predicted,
                extractor=extractor,
            )
            if converted is None:
                missing_label += 1
                continue
            out.write(json.dumps(converted) + "\n")
            written += 1
            if args.progress and idx % 1000 == 0:
                print(f"processed {idx} records (written={written})")

    print(
        json.dumps(
            {
                "input": str(args.fever_json),
                "output": str(args.output),
                "records_written": written,
                "skipped": missing_label,
                "required_pages": len(required_pages),
            },
            indent=2,
        )
    )


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
