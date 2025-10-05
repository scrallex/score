#!/usr/bin/env python3
"""Augment HoVer eval_detail records with secondary-hop evidence.

The HoVer tf-idf doc retrieval results provide a ranked list of pages for each
claim.  We use that list as a lightweight adjacency graph: for every claim we
append the top-N pages that are not already covered by the primary supporting
facts.  Each new page contributes the first sentence from the associated
Wikipedia article together with manifold-aligned metrics so the reliability
model can attend over the expanded memory.

Example:

    python scripts/augment_hover_multi_hop.py \
        --detail results/eval/hover_train/eval_detail.jsonl \
        --hover-split external/hover/data/hover/hover_train_release_v1.1.json \
        --doc-results external/hover/data/hover/tfidf_retrieved/train_tfidf_doc_retrieval_results.json \
        --wiki-db external/hover/wiki_wo_links.db \
        --output results/eval/hover_train_multi_hop/eval_detail.jsonl

The script keeps the original records intact and only appends the newly
retrieved sentences under the ``sentences`` field.  Each appended sentence is
tagged with ``repair_meta.source = "hover_secondary"`` so downstream analysis
can distinguish synthetic hops from gold evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from convert_hover_to_eval import WikiSentenceStore
from eval_feature_utils import FeatureExtractor


@dataclass
class RetrievalResult:
    doc_ids: Sequence[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detail", type=Path, required=True, help="Input eval_detail.jsonl")
    parser.add_argument("--hover-split", type=Path, required=True, help="HoVer split JSON file (train/dev)")
    parser.add_argument("--doc-results", type=Path, required=True, help="TF-IDF doc retrieval results JSON")
    parser.add_argument("--wiki-db", type=Path, required=True, help="Path to wiki_wo_links.db")
    parser.add_argument("--output", type=Path, required=True, help="Augmented eval_detail output path")
    parser.add_argument("--secondary-docs", type=int, default=2, help="Number of secondary pages to append per claim")
    parser.add_argument("--sentences-per-doc", type=int, default=1, help="Sentences to sample from each secondary page")
    parser.add_argument("--min-length", type=int, default=25, help="Skip sentences shorter than this many characters")
    parser.add_argument("--progress", action="store_true", help="Print progress while processing")
    return parser.parse_args()


def load_doc_results(path: Path) -> Dict[str, RetrievalResult]:
    payload = json.loads(path.read_text())
    mapping: Dict[str, RetrievalResult] = {}
    for entry in payload:
        claim_id = entry.get("id")
        doc_results = entry.get("doc_retrieval_results") or []
        if not isinstance(claim_id, str):
            continue
        if not doc_results:
            mapping[claim_id] = RetrievalResult(doc_ids=[])
            continue
        # Each element: [ [docs], [scores], topk ]
        docs = []
        first = doc_results[0]
        if isinstance(first, list) and first:
            docs = [str(doc) for doc in first[0]] if isinstance(first[0], list) else []
        mapping[claim_id] = RetrievalResult(doc_ids=docs)
    return mapping


def load_hover_support(path: Path) -> Dict[str, List[Tuple[str, int]]]:
    payload = json.loads(path.read_text())
    mapping: Dict[str, List[Tuple[str, int]]] = {}
    for entry in payload:
        claim_id = str(entry.get("uid") or entry.get("id"))
        supporting: List[Tuple[str, int]] = []
        for item in entry.get("supporting_facts", []) or []:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            page = str(item[0])
            try:
                idx = int(item[1])
            except (TypeError, ValueError):
                continue
            supporting.append((page, idx))
        mapping[claim_id] = supporting
    return mapping


def normalise_claim_id(record_id: str) -> str:
    token = record_id.rsplit("_", maxsplit=1)[-1]
    return token


def build_sentence_payload(
    text: str,
    *,
    metrics: Dict[str, float],
    citation: str,
    page: str,
    sent_idx: int,
) -> Dict[str, object]:
    return {
        "sentence": text,
        "decisions": {
            "repeat_ok": False,
            "hazard_ok": False,
            "semantic_ok": False,
            "structural_ok": False,
            "admit": False,
        },
        "metrics": metrics,
        "twins": [],
        "action": "decline",
        "repair_span": None,
        "repair_meta": {
            "source": "hover_secondary",
            "page": page,
            "sent_index": sent_idx,
        },
        "citations": [citation],
    }


def augment_records(
    detail_path: Path,
    hover_split: Path,
    doc_results_path: Path,
    wiki_db: Path,
    output_path: Path,
    *,
    secondary_docs: int,
    sentences_per_doc: int,
    min_length: int,
    progress: bool,
) -> None:
    doc_results = load_doc_results(doc_results_path)
    hover_support = load_hover_support(hover_split)
    store = WikiSentenceStore(wiki_db)
    extractor = FeatureExtractor()

    total_records = 0
    augmented_records = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with detail_path.open() as source, output_path.open("w") as sink:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            total_records += 1
            record_id = str(record.get("id") or "")
            claim_id = normalise_claim_id(record_id)
            question = str(record.get("question") or "")
            question_vec = extractor.vector(question)

            existing_pages = {
                str(entry.get("repair_meta", {}).get("page"))
                for entry in record.get("sentences", []) or []
                if isinstance(entry, dict) and isinstance(entry.get("repair_meta"), dict)
                and entry["repair_meta"].get("page")
            }

            doc_candidates = list(doc_results.get(claim_id, RetrievalResult(doc_ids=[])).doc_ids)
            support_pairs = hover_support.get(claim_id, [])
            for page, _ in support_pairs:
                existing_pages.add(page)

            added = 0
            sentences_to_append: List[Dict[str, object]] = []
            for page in doc_candidates:
                if added >= secondary_docs:
                    break
                if not page or page in existing_pages:
                    continue
                for sent_idx in range(sentences_per_doc):
                    sentence = store.sentence(page, sent_idx)
                    if not sentence or len(sentence) < min_length:
                        continue
                    metrics = extractor.metric_vector(sentence, question_vec)
                    citation = f"hover-secondary://{page}#{sent_idx}"
                    sentences_to_append.append(
                        build_sentence_payload(sentence, metrics=metrics, citation=citation, page=page, sent_idx=sent_idx)
                    )
                    break  # only keep first acceptable sentence per page
                if sentences_to_append and sentences_to_append[-1]["repair_meta"]["page"] == page:
                    existing_pages.add(page)
                    added += 1

            if sentences_to_append:
                record.setdefault("sentences", []).extend(sentences_to_append)
                augmented_records += 1

            sink.write(json.dumps(record) + "\n")
            if progress and total_records % 250 == 0:
                print(f"Processed {total_records} records (augmented {augmented_records})")

    store.close()
    if progress:
        print(f"Finished {total_records} records with {augmented_records} augmented entries")


def main() -> None:
    args = parse_args()
    augment_records(
        args.detail,
        args.hover_split,
        args.doc_results,
        args.wiki_db,
        args.output,
        secondary_docs=max(0, args.secondary_docs),
        sentences_per_doc=max(1, args.sentences_per_doc),
        min_length=max(0, args.min_length),
        progress=args.progress,
    )


if __name__ == "__main__":
    main()
