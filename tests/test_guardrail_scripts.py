import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
import calibrate_router
import run_permutation_guardrail as rpg


def _structured_metrics(high_count: int = 4, medium_count: int = 0) -> np.ndarray:
    rows = []
    for _ in range(high_count):
        rows.append((0.95, 0.1, 0.92))
    for _ in range(medium_count):
        rows.append((0.9, 0.2, 0.85))
    for _ in range(10 - high_count - medium_count):
        rows.append((0.5, 0.5, 0.3))
    return np.asarray(rows, dtype=np.float64)


def test_compute_configuration_hits_target_range():
    metrics = _structured_metrics()
    cfg, percentiles, coverage = calibrate_router.compute_configuration(metrics, 0.35, 0.45)

    foreground = cfg["router"]["foreground"]
    assert 0.35 <= coverage <= 0.45
    assert foreground["min_coh"] >= 0.5
    assert pytest.approx(foreground["max_ent"], rel=1e-6) == 0.1
    assert percentiles["coherence"]["P70"] >= 0.9


def test_summarise_domain_reports_expected_metrics(tmp_path):
    domain_root = tmp_path / "planbench"
    metrics_dir = domain_root / "invalid" / "metrics"
    states_dir = domain_root / "invalid" / "states"
    metrics_dir.mkdir(parents=True)
    states_dir.mkdir(parents=True)

    lead_payload = {
        "trace": "sample",
        "failure_index": 3,
        "window_count": 5,
    }
    metrics_path = metrics_dir / "sample.trace_lead.json"
    metrics_path.write_text(json.dumps(lead_payload), encoding="utf-8")

    signals = [
        {"metrics": {"coherence": 0.9, "entropy": 0.1, "stability": 0.8}},
        {"metrics": {"coherence": 0.4, "entropy": 0.6, "stability": 0.2}},
        {"metrics": {"coherence": 0.85, "entropy": 0.12, "stability": 0.75}},
        {"metrics": {"coherence": 0.45, "entropy": 0.55, "stability": 0.3}},
        {"metrics": {"coherence": 0.5, "entropy": 0.5, "stability": 0.4}},
    ]
    state_payload = {"signals": signals}
    state_path = states_dir / "sample.trace_state.json"
    state_path.write_text(json.dumps(state_payload), encoding="utf-8")

    config = {
        "router": {
            "foreground": {
                "min_coh": 0.85,
                "max_ent": 0.12,
                "min_stab": 0.7,
            }
        }
    }
    config_path = tmp_path / "router.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    summary = rpg.summarise_domain(
        domain_root=domain_root,
        config_path=config_path,
        output_path=tmp_path / "summary.json",
        iterations=100,
    )

    assert summary["trace_count"] == 1
    assert summary["coverage_weighted"] == pytest.approx(0.4)
    assert summary["lead_mean"] == pytest.approx(1.0)
    assert summary["p_value_mean"] == pytest.approx(0.88)
    assert summary["precision_mean"] == pytest.approx(1.0)


def test_calibrate_router_dynamic_fallback_triggers(monkeypatch, tmp_path, capsys):
    metrics = []
    metrics.extend([(0.97, 0.1, 0.95)] * 2)
    metrics.extend([(0.92, 0.1, 0.92)] * 2)
    metrics.extend([(0.5, 0.4, 0.4)] * 6)
    signals = [{"metrics": {"coherence": c, "entropy": e, "stability": s}} for c, e, s in metrics]
    state_payload = {"signals": signals}

    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state_payload), encoding="utf-8")

    domain_root = tmp_path / "domain"
    domain_root.mkdir()

    output_path = tmp_path / "router.cfg.json"

    call_history = []

    def fake_summary(*, domain_root, config_path, output_path, iterations):
        if not call_history:
            summary = {
                "coverage_weighted": 0.4,
                "lead_mean": 3.0,
                "p_value_min": 0.2,
                "p_value_mean": 0.2,
            }
        else:
            summary = {
                "coverage_weighted": 0.2,
                "lead_mean": 5.0,
                "p_value_min": 0.03,
                "p_value_mean": 0.03,
            }
        output_path.write_text(json.dumps({"summary": summary}), encoding="utf-8")
        call_history.append({"path": str(config_path), "summary": summary})
        return summary

    monkeypatch.setattr(calibrate_router, "summarise_domain", fake_summary)

    argv = [
        "calibrate_router",
        str(state_path),
        "--target-low",
        "0.35",
        "--target-high",
        "0.45",
        "--output",
        str(output_path),
        "--domain-root",
        str(domain_root),
        "--permutation-iterations",
        "25",
        "--dynamic-target",
        "0.2",
        "--dynamic-window",
        "0.01",
        "--pvalue-threshold",
        "0.05",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    calibrate_router.main()
    captured = capsys.readouterr()

    final_cfg = json.loads(captured.out)
    assert final_cfg["router"]["foreground"]["min_coh"] > 0.9

    coverage_payload = json.loads((output_path.with_suffix(".coverage.json")).read_text(encoding="utf-8"))
    assert coverage_payload["mode"] == "dynamic"
    assert coverage_payload["coverage"] == pytest.approx(0.2)
    assert coverage_payload["base"]["coverage"] == pytest.approx(0.4)
    assert len(call_history) == 2
