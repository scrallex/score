from sep_text_manifold.feedback import suggest_twin_action


def _manifold_state():
    signals = [
        {
            "id": 0,
            "signature": "sigA",
            "metrics": {
                "coherence": 0.9,
                "stability": 0.85,
                "entropy": 0.1,
                "rupture": 0.2,
                "lambda_hazard": 0.2,
            },
        },
        {
            "id": 1,
            "signature": "sigB",
            "metrics": {
                "coherence": 0.7,
                "stability": 0.65,
                "entropy": 0.3,
                "rupture": 0.35,
                "lambda_hazard": 0.35,
            },
        },
        {
            "id": 2,
            "signature": "sigC",
            "metrics": {
                "coherence": 0.4,
                "stability": 0.45,
                "entropy": 0.6,
                "rupture": 0.55,
                "lambda_hazard": 0.55,
            },
        },
    ]
    string_scores = {
        "action__move__ACCEL": {
            "metrics": {
                "coherence": 0.88,
                "stability": 0.82,
                "entropy": 0.12,
                "rupture": 0.22,
            },
            "patternability": 0.91,
            "connector": 0.4,
            "window_ids": [0],
        },
        "action__swap__ACCEL": {
            "metrics": {
                "coherence": 0.68,
                "stability": 0.6,
                "entropy": 0.28,
                "rupture": 0.33,
            },
            "patternability": 0.7,
            "connector": 0.3,
            "window_ids": [1],
        },
    }
    return {"signals": signals, "string_scores": string_scores}


def test_suggest_twin_action_with_window():
    state = _manifold_state()
    invalid = {"window_id": 2, "signature": "sigC"}
    twins = suggest_twin_action(invalid, state, top_k=2)
    assert len(twins) == 2
    assert twins[0].window_id in (2, 1)
    assert all(len(t.metrics) == 5 for t in twins)


def test_suggest_twin_action_signature_filter():
    state = _manifold_state()
    invalid = {"signature": "sigB"}
    twins = suggest_twin_action(invalid, state, match_signature=True)
    assert len(twins) == 1
    assert twins[0].signature == "sigB"


def test_suggest_twin_action_raises_without_context():
    state = _manifold_state()
    try:
        suggest_twin_action({}, state)
    except ValueError:
        return
    raise AssertionError("Expected ValueError when no context provided")
