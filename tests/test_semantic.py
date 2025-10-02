from __future__ import annotations

import numpy as np

from sep_text_manifold.semantic import EmbeddingConfig, SemanticEmbedder, seed_similarity


def test_hash_embeddings_are_deterministic() -> None:
    strings = ["alpha_signal", "beta_signal", "gamma_signal"]
    config = EmbeddingConfig(method="hash", dims=32)
    embedder = SemanticEmbedder(config)
    first = embedder.encode(strings)
    second = embedder.encode(strings)
    assert first.shape == (3, 32)
    assert np.allclose(first, second)


def test_seed_similarity_scores_return_vector() -> None:
    strings = ["alpha_signal", "beta_signal"]
    seeds = ["alpha", "gamma"]
    embedder = SemanticEmbedder(EmbeddingConfig(method="hash", dims=16))
    scores = seed_similarity(strings, embedder=embedder, seeds=seeds)
    assert scores.shape == (2,)
    assert np.isfinite(scores).all()
