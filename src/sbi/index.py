"""Span receipts index built from prepared truth-pack SBI assets."""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import hnswlib
from sentence_transformers import SentenceTransformer

from .span_corpus import load_span_inventory


@dataclass
class StructuralMatch:
    span_id: str
    score: float
    distance: int
    text: str


@dataclass
class SemanticMatch:
    span_id: str
    score: float
    text: str


class SpanReceiptsIndex:
    """Utility that loads span data and exposes membership/twin/context lookups."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        q: int = 3,
        ef_construction: int = 200,
        m: int = 16,
        ef_search: int = 64,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        base_dir = self.manifest_path.parent / "sbi"
        self.span_inventory_path = base_dir / "spans.jsonl"
        self.context_path = base_dir / "contexts.jsonl"
        self.bloom_path = base_dir / "spans.bloom"
        if not self.span_inventory_path.exists():
            raise FileNotFoundError(f"Span inventory missing: {self.span_inventory_path}")
        if not self.bloom_path.exists():
            raise FileNotFoundError(f"Span bloom missing: {self.bloom_path}")

        self.spans: Dict[str, SpanAggregate] = load_span_inventory(self.span_inventory_path)
        self._span_order: List[str] = list(self.spans.keys())
        self._q = q
        self._struct_index = self._build_struct_index(q)
        self._bloom, self._bloom_meta = self._load_bloom()

        self._embedder: Optional[SentenceTransformer] = None
        self._embeddings: Optional[np.ndarray] = None
        self._hnsw: Optional[hnswlib.Index] = None
        self._ef_construction = ef_construction
        self._m = m
        self._ef_search = ef_search
        self._semantic_ready = False
        self._semantic_dir = base_dir
        self._semantic_embeddings_path = base_dir / "semantic_embeddings.npy"
        self._semantic_index_path = base_dir / "semantic.hnsw"
        self._semantic_meta_path = base_dir / "semantic_meta.json"

    # ------------------------------------------------------------------
    def spans_count(self) -> int:
        return len(self.spans)

    # ------------------------------------------------------------------
    def _build_struct_index(self, q: int) -> Dict[str, Dict[str, int]]:
        postings: Dict[str, Dict[str, int]] = {}
        for span_id, span in self.spans.items():
            grams = self._qgrams(span.text, q)
            for gram in grams:
                bucket = postings.setdefault(gram, {})
                bucket[span_id] = bucket.get(span_id, 0) + 1
        return postings

    @staticmethod
    def _qgrams(text: str, q: int) -> List[str]:
        normalised = " " + " ".join(text.lower().split()) + " "
        grams: List[str] = []
        if len(normalised) < q:
            grams.append(normalised)
            return grams
        for idx in range(len(normalised) - q + 1):
            grams.append(normalised[idx : idx + q])
        return grams

    # ------------------------------------------------------------------
    def _load_bloom(self) -> Tuple[object, Dict[str, object]]:
        payload = pickle.loads(self.bloom_path.read_bytes())
        bloom = payload.get("bloom")
        if bloom is None:
            raise ValueError("Bloom payload missing 'bloom' entry")
        meta = {
            "error_rate": payload.get("error_rate", 0.0),
            "count": payload.get("count", len(self.spans)),
        }
        return bloom, meta

    # ------------------------------------------------------------------
    def contains(self, span_id: str) -> bool:
        return span_id in self.spans

    def bloom_contains(self, span_id: str) -> bool:
        return span_id in self._bloom

    def get_span(self, span_id: str) -> Optional[SpanAggregate]:
        return self.spans.get(span_id)

    @property
    def bloom_meta(self) -> Dict[str, object]:
        return dict(self._bloom_meta)

    # ------------------------------------------------------------------
    def structural_search(
        self,
        text: str,
        *,
        top_k: int,
        candidate_multiplier: int = 5,
    ) -> List[StructuralMatch]:
        grams = self._qgrams(text, self._q)
        scores: Dict[str, int] = {}
        for gram in grams:
            postings = self._struct_index.get(gram)
            if not postings:
                continue
            for span_id, freq in postings.items():
                scores[span_id] = scores.get(span_id, 0) + freq
        if not scores:
            return []
        sorted_ids = sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)
        limit = max(top_k * candidate_multiplier, top_k)
        candidates = sorted_ids[:limit]
        matches: List[StructuralMatch] = []
        for span_id, score in candidates:
            span = self.spans.get(span_id)
            if span is None:
                continue
            distance = int(self._levenshtein(text, span.text))
            matches.append(StructuralMatch(span_id=span_id, score=float(score), distance=distance, text=span.text))
        matches.sort(key=lambda m: (m.distance, -m.score))
        return matches[:top_k]

    @staticmethod
    def _levenshtein(a: str, b: str) -> int:
        from rapidfuzz.distance import Levenshtein

        return int(Levenshtein.distance(a, b))

    # ------------------------------------------------------------------
    def _ensure_semantic(self) -> None:
        if self._semantic_ready:
            return
        self._load_or_build_embeddings()
        self._load_or_build_hnsw()
        self._semantic_ready = True

    def _load_or_build_embeddings(self) -> None:
        if self._embeddings is not None:
            return
        if self._semantic_embeddings_path.exists():
            self._embeddings = np.load(self._semantic_embeddings_path)
            return
        self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [self.spans[span_id].text for span_id in self._span_order]
        embeddings = self._embedder.encode(texts, batch_size=128, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        self._embeddings = embeddings.astype(np.float32)
        np.save(self._semantic_embeddings_path, self._embeddings)

    def _load_or_build_hnsw(self) -> None:
        if self._hnsw is not None:
            return
        dim = int(self._embeddings.shape[1])  # type: ignore[union-attr]
        index = hnswlib.Index(space="cosine", dim=dim)
        if self._semantic_index_path.exists() and self._semantic_meta_path.exists():
            index.load_index(str(self._semantic_index_path))
            meta = json.loads(self._semantic_meta_path.read_text())
            ef = int(meta.get("ef", self._ef_search))
            index.set_ef(ef)
            span_order = meta.get("span_order")
            if isinstance(span_order, list) and len(span_order) == len(self._span_order):
                self._span_order = list(span_order)
            self._hnsw = index
            return

        index.init_index(max_elements=len(self._span_order), ef_construction=self._ef_construction, M=self._m)
        labels = np.arange(len(self._span_order))
        index.add_items(self._embeddings, labels)
        index.set_ef(self._ef_search)
        index.save_index(str(self._semantic_index_path))
        meta = {
            "ef": self._ef_search,
            "dim": dim,
            "count": len(self._span_order),
            "span_order": self._span_order,
        }
        self._semantic_meta_path.write_text(json.dumps(meta, indent=2))
        self._hnsw = index

    def semantic_search(self, text: str, *, top_k: int) -> List[SemanticMatch]:
        self._ensure_semantic()
        if self._embedder is None:
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        query_vec = self._embedder.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        labels, distances = self._hnsw.knn_query(query_vec, k=top_k)  # type: ignore[union-attr]
        matches: List[SemanticMatch] = []
        for lbl, dist in zip(labels[0], distances[0]):
            idx = int(lbl)
            span_id = self._span_order[idx]
            span = self.spans[span_id]
            score = float(1.0 - dist)
            matches.append(SemanticMatch(span_id=span_id, score=score, text=span.text))
        return matches

    # ------------------------------------------------------------------
    def contexts(self, span_id: str, side: str, top_k: int) -> List[Dict[str, object]]:
        span = self.spans.get(span_id)
        if span is None:
            return []
        contexts = span.left_contexts if side == "left" else span.right_contexts
        items = sorted(contexts.items(), key=lambda item: (item[1].count, item[0]), reverse=True)[:top_k]
        result = []
        for text, acc in items:
            result.append(
                {
                    "text": text,
                    "count": acc.count,
                    "unique_uris": acc.unique_uris,
                    "unique_domains": acc.unique_domains,
                    "unique_timebins": acc.unique_timebins,
                    "reinforcement": acc.unique_uris + acc.unique_domains,
                }
            )
        return result


__all__ = ["SpanReceiptsIndex", "StructuralMatch", "SemanticMatch"]
