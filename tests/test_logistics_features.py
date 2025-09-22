from scripts.features.logistics_features import build_logistics_features


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
    irreversibility_values = [item["logistics_irreversibility"] for item in features]
    assert irreversibility_values[0] > irreversibility_values[2]
    assert all(0.0 <= value <= 1.0 for value in irreversibility_values)


def test_build_logistics_features_handles_empty():
    state = {"signals": [], "string_scores": {}}
    features = build_logistics_features(state)
    assert features == []

