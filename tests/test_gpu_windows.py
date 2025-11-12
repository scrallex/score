import pytest

from sep_text_manifold import encode
from sep_text_manifold.gpu_windows import HAVE_CUDA, gpu_window_metrics


@pytest.mark.skipif(not HAVE_CUDA, reason="CUDA runtime not available")
def test_gpu_window_metrics_matches_cpu():
    windows = [
        bytes([0xAA] * 8),
        bytes([0xFF] * 8),
        bytes(range(8)),
    ]
    gpu_metrics = gpu_window_metrics(windows, window_bytes=len(windows[0]))
    assert len(gpu_metrics) == len(windows)
    for window, gpu_metric in zip(windows, gpu_metrics):
        cpu_metric = encode.encode_window(window)
        assert pytest.approx(cpu_metric["coherence"], rel=1e-5) == gpu_metric["coherence"]
        assert pytest.approx(cpu_metric["stability"], rel=1e-5) == gpu_metric["stability"]
        assert pytest.approx(cpu_metric["entropy"], rel=1e-5) == gpu_metric["entropy"]
