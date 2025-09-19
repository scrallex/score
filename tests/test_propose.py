import json
from pathlib import Path

from sep_text_manifold.pipeline import analyse_directory
from sep_text_manifold.propose import propose


def test_propose_from_state(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "doc1.txt").write_text("Alpha beta alpha gamma", encoding="utf-8")
    (corpus_dir / "doc2.txt").write_text("Gamma delta beta", encoding="utf-8")

    analysis = analyse_directory(
        str(corpus_dir),
        window_bytes=64,
        stride=32,
        min_token_length=3,
        min_alpha_ratio=0.4,
        drop_numeric=True,
        min_occurrences=1,
        cap_tokens_per_window=20,
    )
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(analysis.to_state(include_profiles=True), indent=2), encoding="utf-8")

    result = propose(state_path, seeds=["alpha"], k=5)
    assert result["proposals"], "Expected non-empty proposals"
