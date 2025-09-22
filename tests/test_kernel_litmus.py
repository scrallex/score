import random
import pytest

try:
    from sep_quantum import analyze_bits
except ImportError:  # pragma: no cover - native module optional in CI
    analyze_bits = None


@pytest.mark.skipif(analyze_bits is None, reason="sep_quantum native module not available")
def test_kernel_litmus():
    rep = analyze_bits([1] * 1024)
    alt = analyze_bits([0, 1] * 512)
    random.seed(1337)
    rnd = analyze_bits([random.getrandbits(1) for _ in range(1024)])

    assert rep.coherence > 0.55 and rep.entropy < 0.20
    assert alt.coherence > 0.75 and alt.entropy < 0.20
    assert rnd.coherence < 0.30 and rnd.entropy > 0.50
