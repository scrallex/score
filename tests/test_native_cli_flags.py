import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest

from sep_text_manifold import native

pytest.importorskip("sep_quantum", reason="sep_quantum native module not available")


@pytest.fixture(autouse=True)
def _restore_native_flag():
    previous = native.use_native()
    native.set_use_native(False)
    try:
        yield
    finally:
        native.set_use_native(previous)


def test_planbench_to_stm_sets_native(monkeypatch, tmp_path):
    from score.scripts import planbench_to_stm

    def fake_collect_traces_from_root(root, domains, traces_dir="traces"):
        return ([tmp_path / "dummy.trace.json"], [])

    def fake_export_traces(mode, paths, adapter, output_root, **kwargs):
        out_path = output_root / f"{mode}_state.json"
        out_path.write_text(json.dumps({"signals": []}), encoding="utf-8")
        return out_path, []

    monkeypatch.setattr(planbench_to_stm, "_collect_traces_from_root", fake_collect_traces_from_root)
    monkeypatch.setattr(planbench_to_stm, "_collect_paths", lambda items: [Path(p) for p in items])
    monkeypatch.setattr(planbench_to_stm, "export_traces", fake_export_traces)
    monkeypatch.setattr(planbench_to_stm, "process_invalid_traces", lambda **kwargs: None)
    monkeypatch.setattr(planbench_to_stm, "PDDLTraceAdapter", lambda *args, **kwargs: object())

    output_dir = tmp_path / "out"
    args = [
        "--input-root",
        str(tmp_path),
        "--domains",
        "logistics",
        "--output",
        str(output_dir),
        "--use-native-quantum",
    ]

    planbench_to_stm.main(args)
    assert native.use_native() is True
    summary_path = output_dir / "run_summary.json"
    assert summary_path.exists()


def test_enrich_features_uses_native_provider(monkeypatch, tmp_path, capsys):
    from score.scripts import enrich_features

    state_path = tmp_path / "state.json"
    tokens_dir = tmp_path / "tokens"
    tokens_dir.mkdir()
    state_payload = {
        "signals": [{"metrics": {"coherence": 0.9, "entropy": 0.1, "stability": 0.8}} for _ in range(2)],
        "settings": {"directory": str(tokens_dir)},
    }
    state_path.write_text(json.dumps(state_payload), encoding="utf-8")

    def fake_build_logistics_features(state, metrics_provider=None):
        return [
            {"logistics_irreversibility": 0.42, "lambda_hazard": 0.12},
            {"logistics_irreversibility": 0.43, "lambda_hazard": 0.11},
        ]

    monkeypatch.setattr(enrich_features, "build_logistics_features", fake_build_logistics_features)

    output_path = tmp_path / "state_enriched.json"
    argv = [
        "enrich_features",
        str(state_path),
        "--features",
        "logistics",
        "--use-native-quantum",
        "--output",
        str(output_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    enrich_features.main()
    captured = json.loads(output_path.read_text(encoding="utf-8"))
    assert native.use_native() is True
    assert captured["signals"][0]["features"]["logistics"]["logistics_irreversibility"] == pytest.approx(0.42)

    stdout = json.loads(capsys.readouterr().out)
    assert stdout["windows"] == 2


def test_calibrate_router_with_native_flag(monkeypatch, tmp_path, capsys):
    from score.scripts import calibrate_router

    state_path = tmp_path / "invalid_state.json"
    state_payload = {
        "signals": [
            {"metrics": {"coherence": 0.95, "entropy": 0.12, "stability": 0.9}},
            {"metrics": {"coherence": 0.92, "entropy": 0.15, "stability": 0.88}},
            {"metrics": {"coherence": 0.5, "entropy": 0.5, "stability": 0.4}},
        ]
    }
    state_path.write_text(json.dumps(state_payload), encoding="utf-8")

    output_path = tmp_path / "router.json"
    argv = [
        "calibrate_router",
        str(state_path),
        "--target-low",
        "0.35",
        "--target-high",
        "0.45",
        "--output",
        str(output_path),
        "--use-native-quantum",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    calibrate_router.main()
    _ = capsys.readouterr()
    assert native.use_native() is True
    assert output_path.exists()
    config_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "router" in config_payload
