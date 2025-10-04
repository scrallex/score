from __future__ import annotations

import json
import pickle
import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pyarrow.parquet as pq
from bloom_filter2 import BloomFilter

from sep_text_manifold.semantic import EmbeddingConfig, SemanticEmbedder


_NORMALISE_RE = re.compile(r"\s+")


def _normalise_span(span: str) -> str:
    return _NORMALISE_RE.sub(" ", span.strip().lower())


def _span_cache_key(span: str) -> str:
    return hashlib.blake2b(span.encode("utf-8"), digest_size=16).hexdigest()


@dataclass
class TwinResult:
    string: str
    occurrences: int
    patternability: float
    semantic_similarity: float
    hazard: float
    source: Optional[str] = None


@dataclass
class SpanMetrics:
    span: str
    norm_span: str
    vector: np.ndarray
    occurrences: int
    patternability: float
    coherence: float
    stability: float
    entropy: float
    rupture: float
    hazard: float
    signature: Optional[str]
    semantic_similarity: float
    twins: List[TwinResult]


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


class TruthPackEngine:
    """Expose manifold lookups and span evaluation for prepared truth-packs."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        seeds: Sequence[str],
        embedding_method: str = "transformer",
        model_name: str = "all-MiniLM-L6-v2",
        hash_dims: int = 256,
        embedding_min_occ: int = 1,
        lru_size: int = 200_000,
    ) -> None:
        manifest = json.loads(manifest_path.read_text())
        self.manifest = manifest
        self.pack_name = manifest.get("name", manifest_path.stem)
        state_path = Path(manifest["state_path"])
        if not state_path.exists():
            raise FileNotFoundError(f"State file not found: {state_path}")
        self.state_path = state_path
        self.state: Dict[str, object] = json.loads(state_path.read_text())
        self.strings: Dict[str, Dict[str, object]] = self.state.get("string_scores", {})  # type: ignore[assignment]
        self.signals: Dict[int, Dict[str, object]] = {
            int(sig["id"]): sig for sig in self.state.get("signals", [])  # type: ignore[list-item]
        }

        self.norm_metrics: Dict[str, Dict[str, object]] = {}
        norm_path = manifest.get("norm_metrics_path")
        if norm_path:
            path = Path(norm_path)
            if not path.is_absolute():
                path = state_path.parent / path
            if path.exists():
                raw_norm = json.loads(path.read_text())
                for key, payload in raw_norm.items():
                    vec = np.array(payload.get("vector", []), dtype=np.float32)
                    payload["vector"] = vec
                    self.norm_metrics[key] = payload

        self.sig_metrics: Dict[bytes, Dict[str, float]] = {}
        signature_table_path = manifest.get("signature_table")
        bloom_path = manifest.get("signature_bloom")
        if signature_table_path:
            table_path = Path(signature_table_path)
            if not table_path.is_absolute():
                table_path = state_path.parent / table_path
            if table_path.exists():
                table = pq.read_table(table_path, memory_map=True)
                sig_col = table.column("sig")
                reps_col = table.column("repetitions")
                lam_col = table.column("lambda")
                coh_col = table.column("coherence")
                stab_col = table.column("stability")
                pattern_col = table.column("patternability")
                for i in range(table.num_rows):
                    sig = sig_col[i].as_buffer().to_pybytes()
                    self.sig_metrics[sig] = {
                        "repetitions": reps_col[i].as_py(),
                        "lambda": lam_col[i].as_py(),
                        "coherence": coh_col[i].as_py(),
                        "stability": stab_col[i].as_py(),
                        "patternability": pattern_col[i].as_py(),
                    }
        if bloom_path:
            path = Path(bloom_path)
            if not path.is_absolute():
                path = state_path.parent / path
            if path.exists():
                with path.open("rb") as fh:
                    self.signature_bloom = pickle.load(fh)
            else:
                self.signature_bloom = None
        else:
            self.signature_bloom = None

        self.seeds = list(seeds)
        self.embedder = SemanticEmbedder(
            EmbeddingConfig(method=embedding_method, model_name=model_name, dims=hash_dims)
        )
        self.seed_vector = self._build_seed_vector()
        self.embedding_min_occ = embedding_min_occ
        self.embedding_strings: List[str] = []
        self.embedding_matrix: Optional[np.ndarray] = None
        self._build_string_embeddings(min_occ=embedding_min_occ)
        self.hazard_cache: Dict[str, float] = {}
        self._metrics_cache = lru_cache(maxsize=lru_size)(self._compute_span_metrics)
        self._sorted_tokens = [
            name
            for name, _ in sorted(
                self.strings.items(), key=lambda kv: int(kv[1].get("occurrences", 0)), reverse=True
            )
        ]

    @classmethod
    def from_manifest(
        cls,
        manifest: Path | str,
        *,
        seeds: Sequence[str],
        embedding_method: str = "transformer",
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
    def _build_seed_vector(self) -> Optional[np.ndarray]:
        if not self.seeds:
            return None
        vectors = self.embedder.encode(self.seeds)
        centroid = vectors.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm == 0.0:
            return np.zeros_like(centroid)
        return centroid / norm

    def _build_string_embeddings(self, min_occ: int) -> None:
        candidates = [name for name, data in self.strings.items() if data.get("occurrences", 0) >= min_occ]
        if not candidates:
            self.embedding_strings = []
            self.embedding_matrix = None
            return
        self.embedding_strings = candidates
        self.embedding_matrix = self.embedder.encode(candidates)

    # ------------------------------------------------------------------
    def semantic_similarity(self, vector: np.ndarray) -> float:
        if self.seed_vector is None:
            return 0.0
        return float(np.clip(np.dot(vector, self.seed_vector), -1.0, 1.0))

    def compute_hazard(self, string: str, data: Optional[Dict[str, object]] = None) -> float:
        norm_key = _normalise_span(string)
        cache_key = norm_key if norm_key in self.hazard_cache else string
        if cache_key in self.hazard_cache:
            return self.hazard_cache[cache_key]
        entry = data or self.strings.get(string) or self.strings.get(norm_key)
        hazards: List[float] = []
        if entry:
            for wid in entry.get("window_ids", []):  # type: ignore[list-item]
                sig = self.signals.get(int(wid))
                if not sig:
                    continue
                metrics = sig.get("metrics", {})
                hazards.append(
                    float(
                        sig.get(
                            "lambda_hazard",
                            metrics.get("lambda_hazard", metrics.get("rupture", 1.0)),
                        )
                    )
                )
        if not hazards and entry:
            metrics = entry.get("metrics", {})
            hazards.append(float(metrics.get("lambda_hazard", metrics.get("rupture", 1.0))))
        hazard_value = float(np.mean(hazards)) if hazards else 1.0
        self.hazard_cache[cache_key] = hazard_value
        return hazard_value

    def top_twins(
        self,
        vector: np.ndarray,
        *,
        exclude: Optional[str] = None,
        limit: int = 3,
    ) -> List[TwinResult]:
        results: List[TwinResult] = []
        seen: set[str] = set()

        if exclude and exclude in self.strings:
            data = self.strings[exclude]
            results.append(
                TwinResult(
                    string=exclude,
                    occurrences=int(data.get("occurrences", 0)),
                    patternability=float(data.get("patternability", 0.0)),
                    semantic_similarity=1.0,
                    hazard=self.compute_hazard(exclude, data),
                )
            )
            seen.add(exclude)

        if self.embedding_matrix is None:
            return results

        scores = self.embedding_matrix @ vector
        order = np.argsort(scores)[::-1]
        for idx in order:
            name = self.embedding_strings[idx]
            if exclude and name == exclude:
                continue
            if name in seen:
                continue
            data = self.strings.get(name)
            if not data:
                continue
            results.append(
                TwinResult(
                    string=name,
                    occurrences=int(data.get("occurrences", 0)),
                    patternability=float(data.get("patternability", 0.0)),
                    semantic_similarity=float(np.clip(scores[idx], -1.0, 1.0)),
                    hazard=self.compute_hazard(name, data),
                )
            )
            seen.add(name)
            if len(results) >= limit:
                break
        return results

    def _span_metrics(self, span: str) -> SpanMetrics:
        norm = _normalise_span(span)
        cached = self.norm_metrics.get(norm)
        if cached:
            vector = cached["vector"]
            if not isinstance(vector, np.ndarray):
                vector = np.array(vector, dtype=np.float32)
                cached["vector"] = vector
            semantic_sim = cached.get("semantic")
            if semantic_sim is None or self.seed_vector is not None:
                if self.seed_vector is not None:
                    semantic_sim = float(np.dot(vector, self.seed_vector))
                else:
                    semantic_sim = 0.0
            return SpanMetrics(
                span=cached.get("string", span),
                norm_span=norm,
                vector=vector,
                occurrences=int(cached.get("occurrences", 0)),
                patternability=float(cached.get("patternability", 0.0)),
                coherence=float(cached.get("coherence", 0.0)),
                stability=float(cached.get("stability", 0.0)),
                entropy=float(cached.get("entropy", 1.0)),
                rupture=float(cached.get("rupture", 0.0)),
                hazard=float(cached.get("hazard", 1.0)),
                signature=cached.get("signature"),
                semantic_similarity=semantic_sim,
                twins=[],
            )
        cache_key = _span_cache_key(norm)
        return self._metrics_cache(cache_key, span, norm)

    def _compute_span_metrics(
        self,
        cache_key: str,
        span: str,
        norm_span: str,
    ) -> SpanMetrics:
        vector = self.embedder.encode([span])[0]
        entry = self.strings.get(span) or self.strings.get(norm_span)
        occurrences = 0
        pattern = 0.0
        coherence = 0.0
        stability = 0.0
        entropy = 1.0
        rupture = 0.0
        signature = None

        if entry:
            occurrences = int(entry.get("occurrences", 0))
            metrics = entry.get("metrics", {})
            pattern = float(entry.get("patternability", metrics.get("coherence", 0.0)))
            coherence = float(metrics.get("coherence", coherence))
            stability = float(metrics.get("stability", stability))
            entropy = float(metrics.get("entropy", entropy))
            rupture = float(metrics.get("rupture", rupture))
            for wid in entry.get("window_ids", []):
                sig = self.signals.get(int(wid))
                if sig and sig.get("signature"):
                    signature = sig.get("signature")
                    break

        hazard = self.compute_hazard(span, entry)
        semantic_similarity = self.semantic_similarity(vector)
        twins: List[TwinResult] = []

        if not entry:
            fallback = None
        else:
            fallback = None
        if not entry and fallback:
            pattern = fallback.patternability
            hazard = fallback.hazard
            fallback_entry = self.strings.get(fallback.string)
            if fallback_entry:
                metrics = fallback_entry.get("metrics", {})
                coherence = float(metrics.get("coherence", coherence))
                stability = float(metrics.get("stability", stability))
                entropy = float(metrics.get("entropy", entropy))
                rupture = float(metrics.get("rupture", rupture))
                for wid in fallback_entry.get("window_ids", []):
                    sig = self.signals.get(int(wid))
                    if sig and sig.get("signature"):
                        signature = sig.get("signature")
                        break

        return SpanMetrics(
            span=span,
            norm_span=norm_span,
            vector=vector,
            occurrences=occurrences,
            patternability=pattern,
            coherence=coherence,
            stability=stability,
            entropy=entropy,
            rupture=rupture,
            hazard=hazard,
            signature=signature,
            semantic_similarity=semantic_similarity,
            twins=twins,
        )

    def cache_clear(self) -> None:
        self.hazard_cache.clear()
        self._metrics_cache.cache_clear()

    def prewarm(self, max_items: int = 20_000) -> None:
        sorted_strings = sorted(
            self.strings.items(), key=lambda item: int(item[1].get("occurrences", 0)), reverse=True
        )
        for name, _ in sorted_strings[: max_items]:
            try:
                self._span_metrics(name)
            except Exception:
                continue

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
        norm = _normalise_span(span)
        sig = hashlib.blake2b(norm.encode("utf-8"), digest_size=16).digest()
        sig_data = self.sig_metrics.get(sig)
        if vector is None and sig_data is not None:
            # try to reuse cached vector from norm metrics
            cached = self.norm_metrics.get(norm)
            if cached:
                vector = cached["vector"] if isinstance(cached["vector"], np.ndarray) else np.array(cached["vector"], dtype=np.float32)

        if self.signature_bloom and sig not in self.signature_bloom and sig_data is None:
            return SpanEvaluation(
                span=span,
                question=question,
                occurrences=0,
                patternability=0.0,
                semantic_similarity=0.0,
                coherence=0.0,
                stability=0.0,
                entropy=1.0,
                rupture=0.0,
                hazard=1.0,
                signature=sig,
                repeat_ok=False,
                hazard_ok=False,
                semantic_ok=False,
                structural_ok=False,
                admitted=False,
                twins=[],
                repair_candidate=None,
            )

        metrics = self._span_metrics(span)
        if vector is None:
            vector = metrics.vector
        sem_score = metrics.semantic_similarity
        occurrences = metrics.occurrences
        hazard = metrics.hazard
        pattern = metrics.patternability
        coherence = metrics.coherence
        stability = metrics.stability
        entropy = metrics.entropy
        rupture = metrics.rupture
        signature = metrics.signature

        if sig_data is not None:
            occurrences = sig_data.get("repetitions", occurrences)
            hazard = sig_data.get("lambda", hazard)
            coherence = sig_data.get("coherence", coherence)
            stability = sig_data.get("stability", stability)
            pattern = sig_data.get("patternability", pattern)
            signature = signature or sig

        twins: List[TwinResult]
        if fetch_twins:
            twins = metrics.twins or self.top_twins(vector, exclude=span if occurrences else None)
        else:
            twins = []
        fallback = twins[0] if twins else None

        if occurrences == 0 and fallback is not None:
            pattern = fallback.patternability
            hazard = fallback.hazard
            base_metrics = self.strings.get(fallback.string)
            if isinstance(base_metrics, dict):
                metrics = base_metrics.get("metrics", {})
                coherence = float(metrics.get("coherence", coherence))
                stability = float(metrics.get("stability", stability))
                entropy = float(metrics.get("entropy", entropy))
                rupture = float(metrics.get("rupture", rupture))
                if not signature:
                    window_ids = base_metrics.get("window_ids", [])
                    for wid in window_ids:
                        sig = self.signals.get(int(wid))
                        if sig and sig.get("signature"):
                            signature = sig.get("signature")
                            break
        if not fetch_twins:
            fallback = None

        repeat_ok = occurrences >= r_min
        hazard_ok = hazard <= hazard_max
        semantic_ok = sem_score >= sigma_min
        structural_ok = pattern >= structural_threshold
        admitted = repeat_ok and hazard_ok and semantic_ok and structural_ok

        return SpanEvaluation(
            span=span,
            question=question,
            occurrences=occurrences,
            patternability=pattern,
            semantic_similarity=sem_score,
            coherence=coherence,
            stability=stability,
            entropy=entropy,
            rupture=rupture,
            hazard=hazard,
            signature=signature,
            repeat_ok=repeat_ok,
            hazard_ok=hazard_ok,
            semantic_ok=semantic_ok,
            structural_ok=structural_ok,
            admitted=admitted,
            twins=twins,
            repair_candidate=fallback if not admitted else None,
        )

    # ------------------------------------------------------------------
    def manifest_path(self) -> Path:
        return Path(self.manifest.get("manifest_path", self.manifest.get("output_root", "")))
