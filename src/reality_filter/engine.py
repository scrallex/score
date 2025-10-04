from __future__ import annotations

import hashlib
import pickle
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq
from bloom_filter2 import BloomFilter
import orjson

try:  # Optional dependency for ANN-backed twin lookup
    import hnswlib  # type: ignore
except ImportError as exc:  # pragma: no cover - gracefully degrade when ANN unavailable
    hnswlib = None  # type: ignore
    _HNSW_IMPORT_ERROR = exc
else:  # pragma: no cover - executed when dependency available
    _HNSW_IMPORT_ERROR = None


_NORMALISE_RE = re.compile(r"\s+")
_HASH_TOKEN_DIGEST_SIZE = 8
_DEFAULT_HASH_DIM = 8192
_ANN_QUERY_K = 12
_TWIN_LIMIT_PER_WINDOW = 2


def _normalise_span(span: str) -> str:
    return _NORMALISE_RE.sub(" ", span.strip().lower())


def _hash_signature(norm_span: str) -> bytes:
    return hashlib.blake2b(norm_span.encode("utf-8"), digest_size=16).digest()


@dataclass
class TwinResult:
    string: str
    occurrences: int
    patternability: float
    semantic_similarity: float
    hazard: float
    source: Optional[str] = None


@dataclass
class SpanEvaluation:
    span: str
    question: Optional[str]
    occurrences: int
    patternability: float
    semantic_similarity: float
    coherence: float
    stability: float
    entropy: float
    rupture: float
    hazard: float
    signature: Optional[str]
    repeat_ok: bool
    hazard_ok: bool
    semantic_ok: bool
    structural_ok: bool
    admitted: bool
    twins: List[TwinResult]
    repair_candidate: Optional[TwinResult]

    def decisions(self) -> Dict[str, bool]:
        return {
            "repeat_ok": self.repeat_ok,
            "hazard_ok": self.hazard_ok,
            "semantic_ok": self.semantic_ok,
            "structural_ok": self.structural_ok,
            "admit": self.admitted,
        }

    def metrics(self) -> Dict[str, float]:
        return {
            "patternability": self.patternability,
            "semantic": self.semantic_similarity,
            "coherence": self.coherence,
            "stability": self.stability,
            "entropy": self.entropy,
            "rupture": self.rupture,
            "lambda": self.hazard,
        }


@dataclass(frozen=True)
class SignatureStats:
    repetitions: int
    lambda_: float
    coherence: float
    stability: float
    patternability: float
    entropy: float
    rupture: float
    window_id: int


class TruthPackEngine:
    """Expose manifold lookups and span evaluation for prepared truth-packs."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        seeds: Sequence[str],
        embedding_method: str = "hash",
        model_name: str = "all-MiniLM-L6-v2",
        hash_dims: int = 256,
        embedding_min_occ: int = 1,
        lru_size: int = 200_000,
    ) -> None:
        del seeds  # seeds handled at pack time via centroids
        del embedding_method, model_name, hash_dims, embedding_min_occ, lru_size

        manifest = orjson.loads(manifest_path.read_bytes())
        self.manifest = manifest
        self.pack_name = manifest.get("name", manifest_path.stem)
        self._manifest_path = manifest_path

        state_path = self._resolve_path(manifest.get("state_path"))
        if not state_path.exists():
            raise FileNotFoundError(f"State file not found: {state_path}")
        self.state_path = state_path
        self.state: Dict[str, object] = orjson.loads(state_path.read_bytes())
        self.strings: Dict[str, Dict[str, object]] = self.state.get("string_scores", {})  # type: ignore[assignment]
        self.signals: Dict[int, Dict[str, object]] = {
            int(sig.get("id", idx)): sig for idx, sig in enumerate(self.state.get("signals", []))  # type: ignore[list-item]
        }

        self._hash_dim = int(manifest.get("hash_dim", _DEFAULT_HASH_DIM))
        self._factual_centroid, self._novelty_centroid = self._load_seed_centroids(manifest.get("seed_centroids"))
        self._margin_cache: Dict[str, float] = {}

        self._signature_stats = self._load_signature_stats(manifest.get("signature_table"))
        self._signature_bloom = self._load_bloom(manifest.get("signature_bloom"))

        self._signature_strings, self._window_strings = self._build_string_indexes()

        self._ann_index, self._ann_meta = self._load_ann_index(
            manifest.get("ann_path"), manifest.get("ann_meta_path")
        )
        self._ann_lock = threading.Lock()

        self._counter_lock = threading.Lock()
        self._counters: Dict[str, float] = {
            "requests_total": 0,
            "admit_true": 0,
            "bloom_hits": 0,
            "bloom_misses": 0,
            "signature_hits": 0,
            "signature_misses": 0,
            "ann_batches": 0,
            "ann_calls": 0,
            "margin_cache_hits": 0,
            "margin_cache_misses": 0,
            "embedder_calls": 0,
            "eval_batches": 0,
            "eval_batch_size_total": 0,
            "eval_time_total_ms": 0,
        }

    # ------------------------------------------------------------------
    @classmethod
    def from_manifest(
        cls,
        manifest: Path | str,
        *,
        seeds: Sequence[str],
        embedding_method: str = "hash",
        model_name: str = "all-MiniLM-L6-v2",
        hash_dims: int = 256,
        embedding_min_occ: int = 1,
        lru_size: int = 200_000,
    ) -> "TruthPackEngine":
        return cls(
            manifest_path=Path(manifest),
            seeds=seeds,
            embedding_method=embedding_method,
            model_name=model_name,
            hash_dims=hash_dims,
            embedding_min_occ=embedding_min_occ,
            lru_size=lru_size,
        )

    # ------------------------------------------------------------------
    def evaluate_span(
        self,
        span: str,
        *,
        question: Optional[str] = None,
        semantic_threshold: float,
        structural_threshold: float,
        r_min: int,
        hazard_max: float,
        sigma_min: float,
        vector: Optional[np.ndarray] = None,
        fetch_twins: bool = False,
    ) -> SpanEvaluation:
        del vector  # retained for backwards compatibility
        evaluations = self.evaluate_spans(
            [span],
            questions=[question] if question is not None else None,
            semantic_threshold=semantic_threshold,
            structural_threshold=structural_threshold,
            r_min=r_min,
            hazard_max=hazard_max,
            sigma_min=sigma_min,
            twins_needed=[fetch_twins],
        )
        return evaluations[0]

    def evaluate_spans(
        self,
        spans: Sequence[str],
        *,
        questions: Optional[Sequence[Optional[str]]] = None,
        semantic_threshold: float,
        structural_threshold: float,
        r_min: int,
        hazard_max: float,
        sigma_min: float,
        twins_needed: bool | Sequence[bool] = False,
    ) -> List[SpanEvaluation]:
        start_time = time.perf_counter()
        if not spans:
            return []

        if isinstance(twins_needed, bool):
            twin_flags = [twins_needed] * len(spans)
        else:
            if len(twins_needed) != len(spans):
                raise ValueError("Length of twins_needed must match spans.")
            twin_flags = list(twins_needed)

        results: List[SpanEvaluation] = [None] * len(spans)  # type: ignore[assignment]
        pending_twins: List[Tuple[int, bytes, SignatureStats, float, str]] = []

        for idx, span in enumerate(spans):
            question = questions[idx] if questions else None
            norm = _normalise_span(span)
            sig = _hash_signature(norm)

            self._bump_counter("requests_total")
            bloom = self._signature_bloom
            if bloom is not None:
                if sig in bloom:
                    self._bump_counter("bloom_hits")
                else:
                    self._bump_counter("bloom_misses")
                    self._bump_counter("signature_misses")
                    results[idx] = self._blocked_evaluation(span, question, sig, norm)
                    continue

            stats = self._signature_stats.get(sig)
            if stats is None:
                self._bump_counter("signature_misses")
                results[idx] = self._blocked_evaluation(span, question, sig, norm)
                continue

            self._bump_counter("signature_hits")

            margin = self._seed_margin_fast(span, norm=norm)
            repeat_ok = stats.repetitions >= r_min
            hazard_ok = stats.lambda_ <= hazard_max
            semantic_ok = margin >= sigma_min and margin >= semantic_threshold
            structural_ok = stats.patternability >= structural_threshold
            admitted = repeat_ok and hazard_ok and semantic_ok and structural_ok

            if admitted:
                self._bump_counter("admit_true")

            evaluation = self._pack_evaluation(
                span=span,
                question=question,
                signature=sig,
                stats=stats,
                margin=margin,
                repeat_ok=repeat_ok,
                hazard_ok=hazard_ok,
                semantic_ok=semantic_ok,
                structural_ok=structural_ok,
                admitted=admitted,
            )
            results[idx] = evaluation

            if admitted or twin_flags[idx]:
                pending_twins.append((idx, sig, stats, margin, norm))

        self._attach_twins(results, pending_twins, limit=3)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self._bump_counter("eval_batches")
        self._bump_counter("eval_batch_size_total", len(spans))
        self._bump_counter("eval_time_total_ms", elapsed_ms)
        return results

    # ------------------------------------------------------------------
    def cache_clear(self) -> None:
        self._margin_cache.clear()

    def prewarm(self, max_items: int = 20_000) -> None:
        warmed = 0
        for entries in self._signature_strings.values():
            for string, *_ in entries:
                _ = self._seed_margin_fast(string)
                warmed += 1
                if warmed >= max_items:
                    return

    def _bump_counter(self, key: str, value: float = 1.0) -> None:
        with self._counter_lock:
            self._counters[key] = self._counters.get(key, 0.0) + value

    def counters_snapshot(self) -> Dict[str, float]:
        with self._counter_lock:
            data = dict(self._counters)
        bloom_total = data.get("bloom_hits", 0.0) + data.get("bloom_misses", 0.0)
        cache_total = data.get("margin_cache_hits", 0.0) + data.get("margin_cache_misses", 0.0)
        requests = data.get("requests_total", 0.0)
        data["bloom_miss_rate"] = (data.get("bloom_misses", 0.0) / bloom_total) if bloom_total else 0.0
        data["cache_hit_rate"] = (data.get("margin_cache_hits", 0.0) / cache_total) if cache_total else 0.0
        data["ann_calls_per_request"] = (data.get("ann_calls", 0.0) / requests) if requests else 0.0
        return data

    def lookup_signature(self, text: str) -> Optional[SignatureStats]:
        norm = _normalise_span(text)
        sig = _hash_signature(norm)
        return self._signature_stats.get(sig)

    def signature_examples(self, text: str, limit: int = 3) -> List[TwinResult]:
        norm = _normalise_span(text)
        sig = _hash_signature(norm)
        return self._direct_twins(sig, self._seed_margin_fast(text, norm=norm), limit)

    def search_strings(self, token: str, limit: int = 3) -> List[str]:
        token_norm = _normalise_span(token)
        if not token_norm:
            return []
        candidates: List[Tuple[int, float, str]] = []
        for string, payload in self.strings.items():
            norm_string = _normalise_span(string)
            if token_norm not in norm_string.split():
                continue
            occurrences = int(payload.get("occurrences", 0))
            metrics = payload.get("metrics", {})
            hazard = float(metrics.get("lambda", metrics.get("rupture", 1.0)))
            candidates.append((occurrences, -hazard, string))
        candidates.sort(reverse=True)
        return [string for _, _, string in candidates[:limit]]

    # ------------------------------------------------------------------
    def _resolve_path(self, path_str: Optional[str]) -> Path:
        if path_str is None:
            raise ValueError("Manifest missing required path entry.")
        path = Path(path_str)
        if path.is_absolute():
            return path
        if path.exists():
            return path
        return (self._manifest_path.parent / path).resolve()

    def _load_seed_centroids(self, centroids_path: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
        factual = np.zeros(self._hash_dim, dtype=np.float32)
        novelty = np.zeros(self._hash_dim, dtype=np.float32)
        if not centroids_path:
            return factual, novelty
        path = self._resolve_path(centroids_path)
        if not path.exists():
            return factual, novelty
        data = np.load(path)
        factual_arr = np.asarray(data.get("factual", factual), dtype=np.float32)
        novelty_arr = np.asarray(data.get("novelty", novelty), dtype=np.float32)
        if factual_arr.size:
            factual = factual_arr
            self._hash_dim = factual_arr.shape[0]
        if novelty_arr.size and novelty_arr.shape[0] == self._hash_dim:
            novelty = novelty_arr
        else:
            novelty = np.zeros(self._hash_dim, dtype=np.float32)
        return factual, novelty

    def _load_signature_stats(self, table_path_str: Optional[str]) -> Dict[bytes, SignatureStats]:
        stats: Dict[bytes, SignatureStats] = {}
        if not table_path_str:
            return stats
        table_path = self._resolve_path(table_path_str)
        if not table_path.exists():
            return stats
        table = pq.read_table(table_path, memory_map=True)
        schema_names = set(table.schema.names)
        sig_col = table.column("sig")
        reps_col = table.column("repetitions").to_numpy(zero_copy_only=False)
        lam_col = table.column("lambda").to_numpy(zero_copy_only=False)
        coh_col = table.column("coherence").to_numpy(zero_copy_only=False)
        stab_col = table.column("stability").to_numpy(zero_copy_only=False)
        pat_col = table.column("patternability").to_numpy(zero_copy_only=False)
        ent_col = (
            table.column("entropy").to_numpy(zero_copy_only=False)
            if "entropy" in schema_names
            else np.full(table.num_rows, 1.0, dtype=np.float32)
        )
        rup_col = (
            table.column("rupture").to_numpy(zero_copy_only=False)
            if "rupture" in schema_names
            else np.zeros(table.num_rows, dtype=np.float32)
        )
        wid_col = (
            table.column("window_id").to_numpy(zero_copy_only=False)
            if "window_id" in schema_names
            else np.zeros(table.num_rows, dtype=np.uint32)
        )
        for idx in range(table.num_rows):
            sig = sig_col[idx].as_buffer().to_pybytes()
            stats[sig] = SignatureStats(
                repetitions=int(reps_col[idx]),
                lambda_=float(lam_col[idx]),
                coherence=float(coh_col[idx]),
                stability=float(stab_col[idx]),
                patternability=float(pat_col[idx]),
                entropy=float(ent_col[idx]),
                rupture=float(rup_col[idx]),
                window_id=int(wid_col[idx]),
            )
        return stats

    def _load_bloom(self, bloom_path_str: Optional[str]) -> Optional[BloomFilter]:
        if not bloom_path_str:
            return None
        path = self._resolve_path(bloom_path_str)
        if not path.exists():
            return None
        with path.open("rb") as fh:
            return pickle.load(fh)

    def _build_string_indexes(self) -> Tuple[Dict[bytes, List[Tuple[str, int, float, float, Optional[int]]]], Dict[int, List[Tuple[str, int, float, float, bytes]]]]:
        signature_map: Dict[bytes, List[Tuple[str, int, float, float, Optional[int]]]] = {}
        window_map: Dict[int, List[Tuple[str, int, float, float, bytes]]] = {}
        for string, payload in self.strings.items():
            norm = _normalise_span(string)
            sig = _hash_signature(norm)
            occurrences = int(payload.get("occurrences", 0))
            metrics = payload.get("metrics", {})
            patternability = float(payload.get("patternability", metrics.get("coherence", 0.0)))
            hazard = float(metrics.get("lambda", metrics.get("rupture", 1.0)))
            sig_stats = self._signature_stats.get(sig)
            if sig_stats is not None:
                hazard = sig_stats.lambda_
            window_ids = [int(wid) for wid in payload.get("window_ids", []) if wid is not None]
            primary_wid = window_ids[0] if window_ids else None
            signature_map.setdefault(sig, []).append((string, occurrences, patternability, hazard, primary_wid))
            for wid in window_ids:
                bucket = window_map.setdefault(wid, [])
                bucket.append((string, occurrences, patternability, hazard, sig))

        for entries in signature_map.values():
            entries.sort(key=lambda item: (item[1], item[2]), reverse=True)
            del entries[8:]
        for entries in window_map.values():
            entries.sort(key=lambda item: (item[1], item[2]), reverse=True)
            del entries[8:]
        return signature_map, window_map

    def _load_ann_index(
        self,
        ann_path_str: Optional[str],
        ann_meta_path_str: Optional[str],
    ) -> Tuple[Optional[object], Optional[Dict[str, object]]]:
        if not ann_path_str or not ann_meta_path_str:
            return None, None
        if hnswlib is None:
            return None, None
        ann_path = self._resolve_path(ann_path_str)
        meta_path = self._resolve_path(ann_meta_path_str)
        if not ann_path.exists() or not meta_path.exists():
            return None, None
        meta = orjson.loads(meta_path.read_bytes())
        dim = int(meta.get("dim", 5))
        index = hnswlib.Index(space="l2", dim=dim)
        index.load_index(str(ann_path))
        index.set_ef(int(meta.get("ef", 100)))
        return index, meta

    # ------------------------------------------------------------------
    def _hash_vector(self, norm_text: str) -> np.ndarray:
        vec = np.zeros(self._hash_dim, dtype=np.float32)
        if not norm_text:
            return vec
        for token in norm_text.split():
            h_bytes = hashlib.blake2b(token.encode("utf-8"), digest_size=_HASH_TOKEN_DIGEST_SIZE).digest()
            idx = int.from_bytes(h_bytes, "little") % self._hash_dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0.0:
            vec /= norm
        return vec

    def _seed_margin_fast(self, text: str, *, norm: Optional[str] = None) -> float:
        norm_span = norm or _normalise_span(text)
        cached = self._margin_cache.get(norm_span)
        if cached is not None:
            self._bump_counter("margin_cache_hits")
            return cached
        self._bump_counter("margin_cache_misses")
        vec = self._hash_vector(norm_span)
        factual = self._factual_centroid
        novelty = self._novelty_centroid
        margin = float(vec @ factual)
        if novelty.size == factual.size and novelty.any():
            margin -= float(vec @ novelty)
        self._margin_cache[norm_span] = margin
        return margin

    def _blocked_evaluation(
        self,
        span: str,
        question: Optional[str],
        sig: bytes,
        norm: Optional[str] = None,
    ) -> SpanEvaluation:
        margin = self._seed_margin_fast(span, norm=norm)
        signature_hex = sig.hex()
        return SpanEvaluation(
            span=span,
            question=question,
            occurrences=0,
            patternability=0.0,
            semantic_similarity=margin,
            coherence=0.0,
            stability=0.0,
            entropy=1.0,
            rupture=0.0,
            hazard=1.0,
            signature=signature_hex,
            repeat_ok=False,
            hazard_ok=False,
            semantic_ok=False,
            structural_ok=False,
            admitted=False,
            twins=[],
            repair_candidate=None,
        )

    def _pack_evaluation(
        self,
        *,
        span: str,
        question: Optional[str],
        signature: bytes,
        stats: SignatureStats,
        margin: float,
        repeat_ok: bool,
        hazard_ok: bool,
        semantic_ok: bool,
        structural_ok: bool,
        admitted: bool,
    ) -> SpanEvaluation:
        return SpanEvaluation(
            span=span,
            question=question,
            occurrences=stats.repetitions,
            patternability=stats.patternability,
            semantic_similarity=margin,
            coherence=stats.coherence,
            stability=stats.stability,
            entropy=stats.entropy,
            rupture=stats.rupture,
            hazard=stats.lambda_,
            signature=signature.hex(),
            repeat_ok=repeat_ok,
            hazard_ok=hazard_ok,
            semantic_ok=semantic_ok,
            structural_ok=structural_ok,
            admitted=admitted,
            twins=[],
            repair_candidate=None,
        )

    def _attach_twins(
        self,
        evaluations: List[SpanEvaluation],
        pending: Sequence[Tuple[int, bytes, SignatureStats, float, str]],
        *,
        limit: int,
    ) -> None:
        if not pending:
            return

        direct_map: Dict[int, List[TwinResult]] = {}
        for idx, sig, _stats, margin, _norm in pending:
            direct_map[idx] = self._direct_twins(sig, margin, limit)

        ann_map = self._ann_twins_batch(pending, direct_map, limit)

        for idx, _sig, _stats, _margin, _norm in pending:
            evaluation = evaluations[idx]
            combined: List[TwinResult] = []
            seen: set[str] = set()

            for twin in direct_map.get(idx, []):
                if twin.string not in seen:
                    combined.append(twin)
                    seen.add(twin.string)

            for twin in ann_map.get(idx, []):
                if twin.string not in seen:
                    combined.append(twin)
                    seen.add(twin.string)
                if len(combined) >= limit:
                    break

            evaluation.twins = combined[:limit]
            if not evaluation.admitted and evaluation.twins:
                evaluation.repair_candidate = evaluation.twins[0]

    def _direct_twins(
        self,
        sig: bytes,
        margin: float,
        limit: int,
    ) -> List[TwinResult]:
        entries = self._signature_strings.get(sig, [])
        twins: List[TwinResult] = []
        for string, occurrences, patternability, hazard, wid in entries[:limit]:
            twins.append(
                TwinResult(
                    string=string,
                    occurrences=occurrences,
                    patternability=patternability,
                    semantic_similarity=margin,
                    hazard=hazard,
                    source=self._make_doc_uri(string, wid),
                )
            )
        return twins[:limit]

    def _ann_metric_vector(self, stats: SignatureStats) -> np.ndarray:
        return np.array(
            [
                stats.coherence,
                stats.stability,
                stats.entropy,
                stats.rupture,
                stats.lambda_,
            ],
            dtype=np.float32,
        )

    def _ann_twins_batch(
        self,
        pending: Sequence[Tuple[int, bytes, SignatureStats, float, str]],
        direct_map: Dict[int, List[TwinResult]],
        limit: int,
    ) -> Dict[int, List[TwinResult]]:
        if self._ann_index is None or not pending:
            return {}
        self._bump_counter("ann_batches")
        self._bump_counter("ann_calls", float(len(pending)))
        vectors = np.vstack([self._ann_metric_vector(stats) for _, _, stats, _, _ in pending])
        try:
            with self._ann_lock:
                labels, distances = self._ann_index.knn_query(vectors, k=_ANN_QUERY_K)
        except Exception:  # pragma: no cover - ANN failures degrade gracefully
            return {}

        ann_results: Dict[int, List[TwinResult]] = {idx: [] for idx, *_ in pending}
        for row, (idx, _sig, _stats, margin, norm) in enumerate(pending):
            seen = {twin.string for twin in direct_map.get(idx, []) if twin.string}
            norm_target = norm
            for lbl, dist in zip(labels[row], distances[row]):
                wid = int(lbl)
                if wid <= 0:
                    continue
                entries = self._window_strings.get(wid, [])
                if not entries:
                    continue
                similarity = max(0.0, 1.0 - float(dist))
                for string, occurrences, patternability, hazard, sig in entries[:_TWIN_LIMIT_PER_WINDOW]:
                    if string in seen:
                        continue
                    if _normalise_span(string) == norm_target:
                        continue
                    ann_results[idx].append(
                        TwinResult(
                            string=string,
                            occurrences=occurrences,
                            patternability=patternability,
                            semantic_similarity=max(margin, similarity),
                            hazard=hazard,
                            source=self._make_doc_uri(string, wid),
                        )
                    )
                    seen.add(string)
                    if len(ann_results[idx]) + len(direct_map.get(idx, [])) >= limit:
                        break
                if len(ann_results[idx]) + len(direct_map.get(idx, [])) >= limit:
                    break
        return ann_results

    def _make_doc_uri(self, string: str, wid: Optional[int]) -> Optional[str]:
        if not string:
            return None
        slug = _normalise_span(string).replace(" ", "_")
        return f"doc://{self.pack_name}#{slug}"


__all__ = ["TruthPackEngine", "SpanEvaluation", "TwinResult"]
