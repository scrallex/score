#!/usr/bin/env python3
"""Convert HoVer multi-hop claims into eval_detail-style records."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

try:
    from nltk.tokenize import sent_tokenize
except LookupError as exc:  # pragma: no cover - runtime tokenizer dependency
    raise RuntimeError(
        "NLTK punkt models are required. Install nltk and run nltk.download('punkt') "
        "and nltk.download('punkt_tab') before invoking convert_hover_to_eval.py"
    ) from exc
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "NLTK is required for sentence tokenisation. Install it via 'pip install nltk'."
    ) from exc

from sep_text_manifold.semantic import EmbeddingConfig, SemanticEmbedder

from eval_feature_utils import FeatureExtractor
from scripts.truth_pack_utils import build_truth_pack_from_texts

LABEL_MAP = {
    "SUPPORTED": "SUPPORTED",
    "NOT_SUPPORTED": "UNVERIFIABLE",
}

BASELINE_LABEL = "UNVERIFIABLE"


def _normalise(vector: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if vector is None:
        return None
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return None
    return vector / norm


@dataclass
class ClaimRecord:
    uid: str
    claim: str
    label: Optional[str]
    supporting_facts: Sequence[Tuple[str, int]]


class WikiSentenceStore:
    """Lazily load and sentence-split Wikipedia articles from the HoVer SQLite dump."""

    def __init__(self, db_path: Path) -> None:
        if not db_path.exists():
            raise FileNotFoundError(f"wiki_wo_links.db not found: {db_path}")
        self._connection = sqlite3.connect(str(db_path))
        self._cache: Dict[str, List[str]] = {}
        self._raw: Dict[str, str] = {}

    def close(self) -> None:
        self._connection.close()

    def _fetch_sentences(self, page: str) -> List[str]:
        normalised = unicodedata.normalize("NFD", page)
        if normalised in self._cache:
            return self._cache[normalised]
        cursor = self._connection.execute(
            "SELECT text FROM documents WHERE id=?",
            (normalised,),
        )
        row = cursor.fetchone()
        if row is None:
            self._raw[normalised] = ""
            sentences = []
        else:
            raw_text = row[0] or ""
            self._raw[normalised] = raw_text
            paragraphs = [segment.strip() for segment in raw_text.split("\n") if segment.strip()]
            sentences = []
            for paragraph in paragraphs:
                try:
                    chunk = sent_tokenize(paragraph)
                except LookupError as exc:  # pragma: no cover - sanity guard for runtime data issues
                    raise RuntimeError(
                        "NLTK punkt models missing at runtime. Run nltk.download('punkt') and "
                        "nltk.download('punkt_tab') before executing convert_hover_to_eval.py"
                    ) from exc
                sentences.extend(token.strip() for token in chunk if token.strip())
        self._cache[normalised] = sentences
        return sentences

    def sentence(self, page: str, index: int) -> Optional[str]:
        sentences = self._fetch_sentences(page)
        if 0 <= index < len(sentences):
            return sentences[index]
        return None

    def article_text(self, page: str) -> Optional[str]:
        sentences = self._fetch_sentences(page)
        if not sentences:
            return None
        return "\n".join(sentences)

    def all_sentences(self, page: str) -> List[str]:
        return list(self._fetch_sentences(page))

    def linked_pages(self, page: str) -> List[str]:
        normalised = unicodedata.normalize("NFD", page)
        if normalised not in self._cache:
            self._fetch_sentences(page)
        raw = self._raw.get(normalised, "")
        if not raw:
            return []
        links = re.findall(r"\[\[([^\]|#]+)", raw)
        result: List[str] = []
        seen: set[str] = set()
        for link in links:
            target = unicodedata.normalize("NFD", link.strip())
            if not target or target == normalised or target in seen:
                continue
            seen.add(target)
            result.append(link.strip())
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hover_json", type=Path, help="Path to HoVer split JSON file (train/dev/test)")
    parser.add_argument("wiki_db", type=Path, help="Path to wiki_wo_links.db used by HoVer")
    parser.add_argument("output", type=Path, help="Destination eval_detail.jsonl output path")
    parser.add_argument("--split", type=str, default="train", help="Split tag for generated ids")
    parser.add_argument("--limit", type=int, help="Optional record limit for debugging")
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
        help="SentenceTransformer model name when using transformer embeddings",
    )
    parser.add_argument(
        "--semantic-dims",
        type=int,
        default=256,
        help="Dimensionality for hash embeddings when transformers are unavailable",
    )
    parser.add_argument(
        "--neighbor-count",
        type=int,
        default=3,
        help="Number of two-hop candidate sentences to append per claim",
    )
    parser.add_argument(
        "--link-limit",
        type=int,
        default=10,
        help="Maximum linked articles explored per gold page",
    )
    parser.add_argument(
        "--truth-pack-root",
        type=Path,
        default=Path("analysis/truth_packs"),
        help="Directory where enriched truth packs will be written",
    )
    parser.add_argument("--progress", action="store_true", help="Emit progress every 500 records")
    return parser.parse_args()


def load_claims(path: Path, limit: Optional[int]) -> Iterable[ClaimRecord]:
    payload = json.loads(path.read_text())
    for idx, entry in enumerate(payload):
        if limit is not None and idx >= limit:
            break
        uid = str(entry.get("uid") or entry.get("id") or idx)
        claim = str(entry.get("claim") or "").strip()
        raw_label = entry.get("label")
        supporting_raw = entry.get("supporting_facts") or []
        supporting: List[Tuple[str, int]] = []
        for item in supporting_raw:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            page = str(item[0])
            try:
                sent_idx = int(item[1])
            except (TypeError, ValueError):
                continue
            supporting.append((page, sent_idx))
        yield ClaimRecord(uid=uid, claim=claim, label=raw_label, supporting_facts=supporting)


def mapped_label(raw: Optional[str]) -> str:
    if raw is None:
        return BASELINE_LABEL
    return LABEL_MAP.get(raw.upper(), BASELINE_LABEL)


def build_sentence(
    text: str,
    *,
    label: str,
    admit: bool,
    metrics: Dict[str, float],
    citation: Optional[str],
    meta: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    decisions = {
        "repeat_ok": admit,
        "hazard_ok": label == "SUPPORTED",
        "semantic_ok": admit,
        "structural_ok": admit,
        "admit": admit,
    }
    payload: Dict[str, object] = {
        "sentence": text,
        "decisions": decisions,
        "metrics": metrics,
        "twins": [],
        "action": "admit" if admit else "decline",
        "repair_span": text if admit else None,
        "repair_meta": meta,
        "citations": [citation] if citation else [],
    }
    return payload


def convert_claim(
    record: ClaimRecord,
    store: WikiSentenceStore,
    extractor: FeatureExtractor,
    split: str,
    *,
    neighbor_count: int,
    link_limit: int,
    corpus_sink: Dict[str, str],
) -> Optional[Dict[str, object]]:
    if not record.claim:
        return None
    label = mapped_label(record.label)
    supported = label == "SUPPORTED"
    question_vec = extractor.vector(record.claim)
    q_vec = _normalise(question_vec)
    claim_metrics = extractor.metric_vector(record.claim, question_vec)

    sentences: List[Dict[str, object]] = []
    sentences.append(
        build_sentence(
            record.claim,
            label=label,
            admit=supported,
            metrics=claim_metrics,
            citation=None,
        )
    )
    corpus_sink.setdefault(f"claim::{record.uid}", record.claim)

    gold_uris: List[str] = []
    record_pages: List[str] = []
    gold_pairs: set[Tuple[str, int]] = set()
    for page, idx in record.supporting_facts:
        text = store.sentence(page, idx)
        if text is None:
            text = f"[missing sentence {idx} in page '{page}']"
        else:
            corpus_sink[f"{page}#{idx}"] = text
            article = store.article_text(page)
            if article:
                corpus_sink.setdefault(f"page::{page}", article)
        record_pages.append(page)
        metrics = extractor.metric_vector(text, question_vec)
        citation = f"hover://{page}#{idx}"
        gold_uris.append(citation)
        meta = {"source": "hover", "page": page, "sent_index": idx}
        sentences.append(
            build_sentence(
                text,
                label=label,
                admit=supported,
                metrics=metrics,
                citation=citation,
                meta=meta,
            )
        )
        gold_pairs.add((page, idx))

    neighbor_candidates: List[Tuple[float, str, int, str]] = []
    if neighbor_count > 0 and q_vec is not None:
        visited_linked: set[str] = set()
        for page in record_pages:
            linked_pages = store.linked_pages(page)
            if link_limit > 0:
                linked_pages = linked_pages[:link_limit]
            for linked in linked_pages:
                if linked in visited_linked or linked in record_pages:
                    continue
                visited_linked.add(linked)
                sentences_linked = store.all_sentences(linked)
                if not sentences_linked:
                    continue
                article = store.article_text(linked)
                if article:
                    corpus_sink.setdefault(f"page::{linked}", article)
                for idx, sent_text in enumerate(sentences_linked):
                    if (linked, idx) in gold_pairs:
                        continue
                    vec = _normalise(extractor.vector(sent_text))
                    if vec is None:
                        continue
                    score = float(np.dot(q_vec, vec))
                    neighbor_candidates.append((score, linked, idx, sent_text))

    if neighbor_candidates:
        neighbor_candidates.sort(key=lambda item: item[0], reverse=True)
        top_candidates = neighbor_candidates[:neighbor_count]
        for rank, (_, page, idx, text) in enumerate(top_candidates, start=1):
            metrics = extractor.metric_vector(text, question_vec)
            citation = f"hover://{page}#{idx}?neighbor=1"
            meta = {"source": "hover_two_hop", "page": page, "sent_index": idx, "rank": rank}
            sentences.append(
                build_sentence(
                    text,
                    label=label,
                    admit=False,
                    metrics=metrics,
                    citation=citation,
                    meta=meta,
                )
            )

    record_id = f"HOVER_{split}_{record.uid}"
    final_answer = record.claim if supported else ""

    detail = {
        "id": record_id,
        "question": record.claim,
        "expected": label,
        "predicted": label,
        "baseline_predicted": BASELINE_LABEL,
        "token": None,
        "final_answer": final_answer,
        "baseline_answer": "",
        "sentences": sentences,
        "hallucinated": not supported,
        "hallucinated_initial": not supported,
        "repaired": False,
        "supported": supported,
        "gold_uris": gold_uris,
        "negative_claim": not supported,
    }
    return detail


def main() -> None:
    args = parse_args()

    embedder_config = EmbeddingConfig(
        method=args.semantic_method,
        model_name=args.semantic_model,
        dims=args.semantic_dims,
    )
    try:
        embedder = SemanticEmbedder(embedder_config)
    except RuntimeError as exc:
        print(f"[convert_hover_to_eval] Failed to initialise semantic embedder: {exc}", file=sys.stderr)
        embedder = SemanticEmbedder(EmbeddingConfig(method="hash", dims=args.semantic_dims))

    extractor = FeatureExtractor(embedder=embedder)
    store = WikiSentenceStore(args.wiki_db)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    records_written = 0
    pages_used: Dict[str, None] = {}
    corpus_sentences: Dict[str, str] = {}
    with args.output.open("w", encoding="utf-8") as handle:
        for idx, claim in enumerate(load_claims(args.hover_json, args.limit), start=1):
            converted = convert_claim(
                claim,
                store,
                extractor,
                args.split,
                neighbor_count=args.neighbor_count,
                link_limit=args.link_limit,
                corpus_sink=corpus_sentences,
            )
            if converted is None:
                continue
            handle.write(json.dumps(converted) + "\n")
            records_written += 1
            for page, _ in claim.supporting_facts:
                pages_used[page] = None
            if args.progress and idx % 500 == 0:
                print(f"processed {idx} claims (written={records_written})")

    text_corpus: Dict[str, str] = dict(corpus_sentences)
    store.close()

    pack_name = f"hover_{args.split}_full"
    manifest_path = build_truth_pack_from_texts(
        pack_name=pack_name,
        texts=text_corpus,
        output_root=args.truth_pack_root / pack_name,
    )

    summary = {
        "input": str(args.hover_json),
        "wiki_db": str(args.wiki_db),
        "output": str(args.output),
        "records": records_written,
        "split": args.split,
        "truth_pack_manifest": str(manifest_path),
        "neighbor_count": args.neighbor_count,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
