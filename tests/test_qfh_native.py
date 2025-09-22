import random

import pytest


sep_quantum = pytest.importorskip("sep_quantum", reason="sep_quantum native module not available")


def test_transform_rich_classification():
    events = sep_quantum.transform_rich([0, 0, 1, 1, 0])
    states = [event.state for event in events]
    assert states == [
        sep_quantum.QFHState.NULL_STATE,
        sep_quantum.QFHState.FLIP,
        sep_quantum.QFHState.RUPTURE,
        sep_quantum.QFHState.FLIP,
    ]
    assert [event.bit_prev for event in events] == [0, 0, 1, 1]
    assert [event.bit_curr for event in events] == [0, 1, 1, 0]


def test_transform_rich_rejects_invalid_bits():
    assert sep_quantum.transform_rich([0, 2, 1]) == []


def test_aggregate_events_groups_consecutive_states():
    events = sep_quantum.transform_rich([0, 1, 1, 0, 0, 1])
    aggregated = sep_quantum.aggregate_events(events)
    states = [entry.state for entry in aggregated]
    counts = [entry.count for entry in aggregated]
    assert states == [
        sep_quantum.QFHState.FLIP,
        sep_quantum.QFHState.RUPTURE,
        sep_quantum.QFHState.FLIP,
        sep_quantum.QFHState.NULL_STATE,
        sep_quantum.QFHState.FLIP,
    ]
    assert counts == [1, 1, 1, 1, 1]


def test_analyze_window_constant_zero_stream():
    window = [0] * 32
    result = sep_quantum.analyze_window(window)
    assert result.null_state_count == len(window) - 1
    assert result.flip_count == 0
    assert result.rupture_count == 0
    assert result.rupture_ratio == pytest.approx(0.0)
    assert result.entropy == pytest.approx(0.05, abs=1e-6)
    assert result.coherence == pytest.approx(0.67900002, rel=1e-4)
    assert result.collapse_detected is False


def test_analyze_window_alternating_stream():
    window = [0, 1] * 16
    result = sep_quantum.analyze_window(window)
    assert result.null_state_count == 0
    assert result.flip_count == len(window) - 1
    assert result.rupture_count == 0
    assert result.flip_ratio == pytest.approx(1.0)
    assert result.entropy == pytest.approx(0.05, abs=1e-6)
    assert result.coherence == pytest.approx(0.90786785, rel=1e-4)
    assert result.collapse_detected is False


def test_analyze_window_all_ones_hazard_and_collapse():
    window = [1] * 32
    result = sep_quantum.analyze_window(window)
    assert result.null_state_count == 0
    assert result.flip_count == 0
    assert result.rupture_count == len(window) - 1
    assert result.rupture_ratio == pytest.approx(1.0)
    assert result.entropy == pytest.approx(0.05, abs=1e-6)
    # Coherence is damped by the heavy rupture regime.
    assert result.coherence == pytest.approx(0.59683347, rel=1e-4)
    assert result.collapse_detected is True


def test_analyze_window_random_noise_near_equal_distribution():
    random.seed(1337)
    window = [random.getrandbits(1) for _ in range(64)]
    result = sep_quantum.analyze_window(window)
    assert 0 < result.null_state_count < len(window)
    assert 0 < result.flip_count < len(window)
    assert 0 < result.rupture_count < len(window)
    assert result.entropy > 0.4
    assert result.coherence < 0.5


def test_native_wrappers_expose_events_and_results():
    from sep_text_manifold import native

    if not native.HAVE_NATIVE:
        pytest.skip("sep_quantum bindings not built")

    bits = [0, 1, 1, 0, 1, 1, 1, 0]
    result = native.analyze_window(bits)
    assert isinstance(result, native.QFHResult)
    assert result.flip_count > 0

    events = native.transform_rich(bits)
    assert events
    assert all(isinstance(evt, native.QFHEvent) for evt in events)

    aggregates = native.aggregate_events(events)
    assert aggregates
    assert all(isinstance(entry, native.QFHAggregateEvent) for entry in aggregates)


def test_encode_window_matches_native_and_fallback(monkeypatch):
    from sep_text_manifold import encode, native

    if not native.HAVE_NATIVE:
        pytest.skip("sep_quantum bindings not built")

    random.seed(42)
    window = bytes(random.getrandbits(8) for _ in range(32))
    bits = encode.bytes_to_bits(window)

    native_result = native.analyze_window(bits)
    native_metrics = encode.encode_window(window)
    rupture_ratio = float(native_result.rupture_ratio)
    assert native_metrics["coherence"] == pytest.approx(float(native_result.coherence))
    assert native_metrics["stability"] == pytest.approx(1.0 - rupture_ratio)
    assert native_metrics["entropy"] == pytest.approx(float(native_result.entropy))
    assert native_metrics["rupture"] == pytest.approx(rupture_ratio)
    assert native_metrics["lambda_hazard"] == pytest.approx(rupture_ratio)

    original_flag = native.HAVE_NATIVE
    monkeypatch.setattr(native, "HAVE_NATIVE", False)
    try:
        fallback_metrics = encode.encode_window(window)
    finally:
        monkeypatch.setattr(native, "HAVE_NATIVE", original_flag)

    expected_fallback = encode.compute_metrics(bits)
    for key, value in expected_fallback.items():
        assert fallback_metrics[key] == pytest.approx(value)
