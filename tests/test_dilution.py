from sep_text_manifold.dilution import (
    compute_dilution_metrics,
    path_dilution,
    semantic_dilution,
    signal_dilution,
)


def test_path_dilution_entropy():
    signatures = ["a", "b", "a", "c"]
    value = path_dilution(signatures, 0)
    assert 0.9 < value <= 1.0
    assert path_dilution(signatures, 1) == 0.0
    assert path_dilution(signatures, 3) == 0.0


def test_signal_dilution_diversity():
    assert signal_dilution(["foo", "foo"]) == 0.0
    diverse = signal_dilution(["foo", "bar", "baz"])
    assert 0.9 < diverse <= 1.0


def test_semantic_dilution_alignment():
    structural = ["sig_a", "sig_b", "sig_a", "sig_c"]
    semantic = ["alpha", "beta", "alpha", "gamma"]
    aligned = semantic_dilution(structural, semantic)
    assert 0.0 <= aligned < 0.5
    misaligned = semantic_dilution(structural, ["omega"] * len(structural))
    assert misaligned > aligned
    assert misaligned == 1.0


def test_compute_dilution_metrics_series():
    signals = [
        {"id": 0, "signature": "a"},
        {"id": 1, "signature": "b"},
        {"id": 2, "signature": "a"},
        {"id": 3, "signature": "c"},
    ]
    token_windows = {
        "alpha": {"window_ids": [0, 2]},
        "beta": {"window_ids": [0, 1]},
        "gamma": {"window_ids": [3]},
    }
    path_series, signal_series, semantic_score = compute_dilution_metrics(signals, token_windows)
    assert len(path_series) == len(signals)
    assert len(signal_series) == len(signals)
    assert 0.0 <= semantic_score <= 1.0
    assert any(value > 0.0 for value in signal_series)
