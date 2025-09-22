import json
from pathlib import Path

import pytest

from scripts.adapters import RealWorldAdapter
from scripts.enrich_features import enrich_with_causal_features
from scripts.features import CausalFeatureExtractor


def test_causal_feature_extractor_scales_outputs():
    extractor = CausalFeatureExtractor()
    history = [
        {
            "metrics": {"coherence": 0.6, "entropy": 0.4, "stability": 0.8},
            "dilution": {"path": 0.2, "signal": 0.3},
            "state": {"resources": {"truck": {"locked": 1}}},
        }
    ]
    window = {
        "metrics": {"coherence": 0.3, "entropy": 0.7, "stability": 0.5},
        "dilution": {"path": 0.4, "signal": 0.6},
        "state": {"resources": {"truck": {"locked": 2}}},
    }

    features = extractor.extract(window, history=history)
    assert set(features) == {
        "irreversible_actions",
        "resource_commitment_ratio",
        "decision_reversibility",
        "unsatisfied_preconditions",
        "effect_cascade_depth",
        "constraint_violation_distance",
        "action_velocity",
        "state_divergence_rate",
        "pattern_break_score",
    }
    assert all(0.0 <= value <= 1.0 for value in features.values())


def test_enrich_with_causal_features(tmp_path):
    state = {
        "signals": [
            {"metrics": {"coherence": 0.5, "entropy": 0.4, "stability": 0.9}},
            {"metrics": {"coherence": 0.2, "entropy": 0.8, "stability": 0.6}},
        ]
    }
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state), encoding="utf-8")

    loaded = json.loads(path.read_text(encoding="utf-8"))
    updated = enrich_with_causal_features(loaded)

    assert updated == 2
    for window in loaded["signals"]:
        assert "causal" in window["features"]
        assert all(0.0 <= v <= 1.0 for v in window["features"]["causal"].values())


def test_real_world_adapter_from_github_actions(tmp_path):
    workflow_payload = {
        "workflow_runs": [
            {
                "id": 101,
                "name": "deploy",
                "html_url": "https://example.org/run/101",
                "steps": [
                    {"name": "lint", "conclusion": "success", "status": "completed", "duration_ms": 12000},
                    {"name": "deploy", "conclusion": "failure", "status": "completed", "duration_ms": 45000},
                ],
            }
        ]
    }
    run_path = tmp_path / "runs.json"
    run_path.write_text(json.dumps(workflow_payload), encoding="utf-8")

    adapter = RealWorldAdapter()
    state = adapter.from_github_actions(run_path)

    assert state["metadata"]["domain"] == "github_actions"
    assert state["failure_index"] == 1
    signals = state["signals"]
    assert len(signals) == 2
    assert signals[1]["failure"] is True
    assert "causal" in signals[0]["features"]
    assert signals[0]["metrics"]["coherence"] <= 1.0
