from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Optional

import numpy as np

from sep_text_manifold.encode import encode_window
from sep_text_manifold.scoring import patternability_score
from sep_text_manifold.semantic import SemanticEmbedder

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def structural_metrics(text: str) -> Dict[str, float]:
    payload = text.encode("utf-8", "ignore")
    if not payload:
        return {
            "coherence": 0.0,
            "stability": 0.0,
            "entropy": 0.0,
            "rupture": 0.0,
            "lambda": 0.0,
        }
    metrics = encode_window(payload)
    return {
        "coherence": float(metrics.get("coherence", 0.0)),
        "stability": float(metrics.get("stability", 0.0)),
        "entropy": float(metrics.get("entropy", 0.0)),
        "rupture": float(metrics.get("rupture", 0.0)),
        "lambda": float(metrics.get("lambda_hazard", metrics.get("rupture", 0.0))),
    }


def repetition_count(text: str) -> int:
    tokens = TOKEN_PATTERN.findall(text.lower())
    if not tokens:
        return 0
    counts = Counter(tokens)
    return sum(count for count in counts.values() if count > 1)


class FeatureExtractor:
    """Compute manifold-aligned feature vectors for textual spans."""

    def __init__(self, embedder: Optional[SemanticEmbedder] = None) -> None:
        self.embedder = embedder or SemanticEmbedder()
        self._vector_cache: Dict[str, np.ndarray] = {}

    def vector(self, text: str) -> Optional[np.ndarray]:
        key = text.strip()
        if not key:
            return None
        cached = self._vector_cache.get(key)
        if cached is not None:
            return cached
        vector = self.embedder.encode([key])[0]
        self._vector_cache[key] = vector
        return vector

    def semantic_similarity(self, reference: Optional[np.ndarray], text: str) -> float:
        if reference is None:
            return 0.0
        vector = self.vector(text)
        if vector is None:
            return 0.0
        sim = float(np.clip(np.dot(reference, vector), -1.0, 1.0))
        return 0.5 * (sim + 1.0)

    def metric_vector(self, text: str, reference: Optional[np.ndarray]) -> Dict[str, float]:
        structural = structural_metrics(text)
        patternability = float(
            patternability_score(
                structural["coherence"],
                structural["stability"],
                structural["entropy"],
                structural["rupture"],
            )
        )
        semantic = self.semantic_similarity(reference, text)
        return {
            "patternability": patternability,
            "semantic": semantic,
            "coherence": structural["coherence"],
            "stability": structural["stability"],
            "entropy": structural["entropy"],
            "rupture": structural["rupture"],
            "lambda": structural["lambda"],
            "repetitions": float(repetition_count(text)),
        }
