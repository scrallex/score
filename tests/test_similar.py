import json

from sep_text_manifold.similar import cross_corpus_similarity
from sep_text_manifold import cli


def _build_source_state() -> dict:
    return {
        "string_scores": {
            "alpha": {
                "metrics": {
                    "coherence": 0.9,
                    "stability": 0.84,
                    "entropy": 0.12,
                    "rupture": 0.18,
                    "lambda_hazard": 0.18,
                },
                "patternability": 0.72,
                "connector": 0.55,
                "occurrences": 4,
                "window_ids": [1, 2],
            },
            "beta": {
                "metrics": {
                    "coherence": 0.6,
                    "stability": 0.58,
                    "entropy": 0.35,
                    "rupture": 0.42,
                },
                "patternability": 0.35,
                "connector": 0.20,
                "occurrences": 2,
                "window_ids": [3],
            },
        }
    }


def _build_target_state() -> dict:
    return {
        "signals": [
            {
                "id": 10,
                "window_start": 0,
                "window_end": 128,
                "signature": "sig_a",
                "metrics": {
                    "coherence": 0.88,
                    "stability": 0.82,
                    "entropy": 0.15,
                    "rupture": 0.19,
                },
                "lambda_hazard": 0.19,
            },
            {
                "id": 11,
                "window_start": 64,
                "window_end": 192,
                "signature": "sig_b",
                "metrics": {
                    "coherence": 0.55,
                    "stability": 0.50,
                    "entropy": 0.40,
                    "rupture": 0.45,
                },
                "lambda_hazard": 0.45,
            },
        ]
    }


def test_cross_corpus_similarity_fallback():
    source_state = _build_source_state()
    target_state = _build_target_state()

    result = cross_corpus_similarity(
        source_state,
        profile="coh>=0.8",
        min_connector=0.5,
        min_patternability=0.6,
        target_state=target_state,
        k=2,
    )

    assert result["candidate_count"] == 1
    entry = result["results"][0]
    assert entry["string"] == "alpha"
    matches = entry["matches"]
    assert matches
    assert matches[0]["window_id"] == 10
    assert matches[0]["metrics"]["coherence"] == 0.88


def test_cli_similar_fallback(tmp_path, capsys):
    source_state = _build_source_state()
    target_state = _build_target_state()

    source_path = tmp_path / "source_state.json"
    target_path = tmp_path / "target_state.json"
    source_path.write_text(json.dumps(source_state), encoding="utf-8")
    target_path.write_text(json.dumps(target_state), encoding="utf-8")

    cli.main(
        [
            "similar",
            "--source",
            str(source_path),
            "--target-state",
            str(target_path),
            "--profile",
            "coh>=0.8",
            "--min-connector",
            "0.5",
            "--min-patternability",
            "0.6",
            "--limit",
            "1",
            "--k",
            "1",
        ]
    )

    out = capsys.readouterr().out
    assert "candidate_count" in out
    assert "alpha" in out


def test_percentile_profile():
    source_state = _build_source_state()
    target_state = _build_target_state()

    result = cross_corpus_similarity(
        source_state,
        profile="coh>=P0",
        min_connector=0.5,
        min_patternability=0.6,
        target_state=target_state,
        k=2,
    )

    assert result["candidate_count"] == 1
