#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

#include "sep_core/core/qfh.h"

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

Metrics analyze_bits_native(const std::vector<std::uint8_t>& bits) {
    using namespace sep::quantum;
    static QFHOptions options;
    static QFHBasedProcessor processor(options);
    auto result = processor.analyze(bits);
    const float coherence = static_cast<float>(result.coherence);
    const float entropy = static_cast<float>(result.entropy);
    const float rupture = static_cast<float>(result.rupture_ratio);
    const float stability = 1.0f - rupture;
    Metrics metrics{};
    metrics.coherence = coherence;
    metrics.stability = stability;
    metrics.entropy = entropy;
    metrics.rupture = rupture;
    metrics.lambda_hazard = rupture;
    metrics.sig_c = bucket(coherence);
    metrics.sig_s = bucket(stability);
    metrics.sig_e = bucket(entropy);
    return metrics;
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

    m.def("analyze_bits", &analyze_bits_native, "Analyze a window of bits using the native QFH/QBSA kernel");
}
