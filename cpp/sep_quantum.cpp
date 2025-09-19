#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "manifold_record.h"

// Placeholder includes for the real SEP implementation.
// #include "qfh.h"
// #include "qbsa.h"

namespace py = pybind11;

struct Metrics {
    float coherence;
    float stability;
    float entropy;
    float rupture;
    float lambda_hazard;
    std::uint16_t sig_c;
    std::uint16_t sig_s;
    std::uint16_t sig_e;
};

static std::uint16_t bucket(float value) {
    value = std::fmax(0.0f, std::fmin(1.0f, value));
    return static_cast<std::uint16_t>(std::lround(value * 1000.0f));
}

Metrics analyze_bits(const std::vector<std::uint8_t>& bits) {
    // TODO: replace this stub with the actual QFH/QBSA computation.
    // The placeholder below preserves the Python fallback semantics.
    const std::size_t n = bits.size();
    if (n == 0) {
        return {0.f, 0.f, 0.f, 0.f, 0.f, 0, 0, 0};
    }
    std::size_t ones = 0;
    for (auto bit : bits) {
        if (bit) {
            ++ones;
        }
    }
    float p_one = static_cast<float>(ones) / static_cast<float>(n);
    float p_zero = 1.0f - p_one;
    auto entropy_component = [](float p) -> float {
        return (p > 0.0f) ? -p * std::log2(p) : 0.0f;
    };
    float entropy = entropy_component(p_zero) + entropy_component(p_one);
    if (entropy > 1.0f) {
        entropy = 1.0f;
    }
    // Very rough stand-ins mirroring encode.compute_metrics.
    float coherence = 1.0f - entropy;
    std::size_t transitions = 0;
    for (std::size_t i = 1; i < n; ++i) {
        if (bits[i] != bits[i - 1]) {
            ++transitions;
        }
    }
    float rupture = (n > 1) ? static_cast<float>(transitions) / static_cast<float>(n - 1) : 0.0f;
    float stability = 1.0f - rupture;
    float lambda_hazard = rupture;
    return {
        coherence,
        stability,
        entropy,
        rupture,
        lambda_hazard,
        bucket(coherence),
        bucket(stability),
        bucket(entropy),
    };
}

PYBIND11_MODULE(sep_quantum, m) {
    py::class_<Metrics>(m, "Metrics")
        .def_readonly("coherence", &Metrics::coherence)
        .def_readonly("stability", &Metrics::stability)
        .def_readonly("entropy", &Metrics::entropy)
        .def_readonly("rupture", &Metrics::rupture)
        .def_readonly("lambda_hazard", &Metrics::lambda_hazard)
        .def_readonly("sig_c", &Metrics::sig_c)
        .def_readonly("sig_s", &Metrics::sig_s)
        .def_readonly("sig_e", &Metrics::sig_e);

    m.def("analyze_bits", &analyze_bits, "Analyze a window of bits and return manifold metrics");
}
