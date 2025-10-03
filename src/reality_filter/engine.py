from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sep_text_manifold.semantic import EmbeddingConfig, SemanticEmbedder


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
    admitted: bool
    twins: List[TwinResult]
    repair_candidate: Optional[TwinResult]

    def decisions(self) -> Dict[str, bool]:
        return {
            "repeat_ok": self.repeat_ok,
            "hazard_ok": self.hazard_ok,
            "semantic_ok": self.semantic_ok,
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
    ) -> "TruthPackEngine":
        return cls(
            manifest_path=Path(manifest),
            seeds=seeds,
            embedding_method=embedding_method,
            model_name=model_name,
            hash_dims=hash_dims,
            embedding_min_occ=embedding_min_occ,
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
        if string in self.hazard_cache:
            return self.hazard_cache[string]
        entry = data or self.strings.get(string)
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
        self.hazard_cache[string] = hazard_value
        return hazard_value

    def metrics_for_string(
        self,
        string: str,
        data: Optional[Dict[str, object]] = None,
    ) -> Tuple[float, float, float, float, float]:
        entry = data or self.strings.get(string)
        if entry:
            metrics = entry.get("metrics", {})
            return (
                float(entry.get("patternability", 0.0)),
                float(metrics.get("coherence", 0.0)),
                float(metrics.get("stability", 0.0)),
                float(metrics.get("entropy", 1.0)),
                float(metrics.get("rupture", metrics.get("rupture", 0.0))),
            )
        return 0.0, 0.0, 0.0, 1.0, 0.0

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
    ) -> SpanEvaluation:
        if vector is None:
            vector = self.embedder.encode([span])[0]
        sem_score = self.semantic_similarity(vector)
        data = self.strings.get(span)
        occurrences = int(data.get("occurrences", 0)) if data else 0
        hazard = self.compute_hazard(span, data)
        pattern, coherence, stability, entropy, rupture = self.metrics_for_string(span, data)

        signature = None
        if data:
            window_ids = data.get("window_ids", [])
            for wid in window_ids:
                sig = self.signals.get(int(wid))
                if sig and sig.get("signature"):
                    signature = sig.get("signature")
                    break

        twins = self.top_twins(vector, exclude=span, limit=3)
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

        repeat_ok = occurrences >= r_min
        hazard_ok = hazard <= hazard_max
        semantic_ok = sem_score >= sigma_min
        admitted = repeat_ok and hazard_ok and semantic_ok

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
            admitted=admitted,
            twins=twins,
            repair_candidate=fallback if not admitted else None,
        )

    # ------------------------------------------------------------------
    def manifest_path(self) -> Path:
        return Path(self.manifest.get("manifest_path", self.manifest.get("output_root", "")))
