import pytest

from scripts.features.logistics_features import build_logistics_features
from sep_text_manifold.encode import bytes_to_bits
from sep_text_manifold import native


def test_build_logistics_features_basic_distribution():
    state = {
        "signals": [{} for _ in range(3)],
        "string_scores": {
            "deliver": {"window_ids": [0]},
            "unload": {"window_ids": [1]},
            "load": {"window_ids": [2]},
            "not_at_pkg1": {"window_ids": [0, 2]},
            "at_pkg1": {"window_ids": [1]},
        },
    }

    features = build_logistics_features(state)
    assert len(features) == 3
    for item in features:
        expected_keys = {
            "logistics_irreversibility",
            "logistics_momentum",
            "logistics_cluster_entropy",
            "logistics_predicate_balance",
            "logistics_predicate_delta",
            "coherence",
            "stability",
            "entropy",
            "rupture",
            "lambda_hazard",
            "signature",
        }
        assert expected_keys.issubset(item.keys())
        assert all(0.0 <= float(item[key]) <= 1.0 for key in expected_keys if key != "signature")

    irreversibility_values = [item["logistics_irreversibility"] for item in features]
    assert irreversibility_values[0] > irreversibility_values[2]


def test_build_logistics_features_handles_empty():
    state = {"signals": [], "string_scores": {}}
    features = build_logistics_features(state)
    assert features == []


def test_build_logistics_features_native_provider_matches_metrics():
    pytest.importorskip("sep_quantum")
    if not native.HAVE_NATIVE:
        pytest.skip("Native bindings not available")

    base_state = {
        "signals": [
            {"window_bytes": b"AAAAAA"},
            {"window_bytes": b"BBBBBB"},
        ],
        "string_scores": {
            "deliver": {"window_ids": [0]},
            "load": {"window_ids": [1]},
            "not_at_pkg1": {"window_ids": [0]},
        },
    }

    def provider(window):
        payload = window.get("window_bytes")
        bits = list(bytes_to_bits(payload)) if isinstance(payload, (bytes, bytearray)) else []
        if not bits:
            return {}
        result = native.analyze_window(bits)
        return {
            "coherence": float(result.coherence),
            "entropy": float(result.entropy),
            "rupture": float(result.rupture_ratio),
            "stability": float(getattr(result, "stability", 1.0 - float(result.rupture_ratio))),
            "lambda_hazard": float(result.rupture_ratio),
        }

    features = build_logistics_features(base_state, metrics_provider=provider)
    for signal, feature in zip(base_state["signals"], features):
        payload = signal["window_bytes"]
        bits = list(bytes_to_bits(payload))
        result = native.analyze_window(bits)
        assert feature["coherence"] == pytest.approx(float(result.coherence))
        assert feature["entropy"] == pytest.approx(float(result.entropy))
        assert feature["rupture"] == pytest.approx(float(result.rupture_ratio))
        assert feature["stability"] == pytest.approx(float(result.stability))
        assert feature["lambda_hazard"] == pytest.approx(float(result.rupture_ratio))
