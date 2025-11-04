#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdint>
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
    processor.reset();
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

sep::quantum::QFHResult analyze_bits_detailed(const std::vector<std::uint8_t>& bits) {
    using namespace sep::quantum;
    static QFHOptions options;
    static QFHBasedProcessor processor(options);
    processor.reset();
    return processor.analyze(bits);
}

PYBIND11_MODULE(sep_quantum, m) {
    py::enum_<sep::quantum::QFHState>(m, "QFHState")
        .value("NULL_STATE", sep::quantum::QFHState::NULL_STATE)
        .value("STABLE", sep::quantum::QFHState::STABLE)
        .value("UNSTABLE", sep::quantum::QFHState::UNSTABLE)
        .value("COLLAPSING", sep::quantum::QFHState::COLLAPSING)
        .value("COLLAPSED", sep::quantum::QFHState::COLLAPSED)
        .value("RECOVERING", sep::quantum::QFHState::RECOVERING)
        .value("FLIP", sep::quantum::QFHState::FLIP)
        .value("RUPTURE", sep::quantum::QFHState::RUPTURE)
        .export_values();

    py::class_<sep::quantum::QFHEvent>(m, "QFHEvent")
        .def_readonly("index", &sep::quantum::QFHEvent::index)
        .def_readonly("state", &sep::quantum::QFHEvent::state)
        .def_readonly("bit_prev", &sep::quantum::QFHEvent::bit_prev)
        .def_readonly("bit_curr", &sep::quantum::QFHEvent::bit_curr);

    py::class_<sep::quantum::QFHAggregateEvent>(m, "QFHAggregateEvent")
        .def_readonly("index", &sep::quantum::QFHAggregateEvent::index)
        .def_readonly("state", &sep::quantum::QFHAggregateEvent::state)
        .def_readonly("count", &sep::quantum::QFHAggregateEvent::count);

    py::class_<sep::quantum::QFHResult>(m, "QFHResult")
        .def_readonly("coherence", &sep::quantum::QFHResult::coherence)
        .def_readonly("stability", &sep::quantum::QFHResult::stability)
        .def_readonly("confidence", &sep::quantum::QFHResult::confidence)
        .def_readonly("collapse_detected", &sep::quantum::QFHResult::collapse_detected)
        .def_readonly("rupture_ratio", &sep::quantum::QFHResult::rupture_ratio)
        .def_readonly("final_state", &sep::quantum::QFHResult::final_state)
        .def_readonly("collapse_threshold", &sep::quantum::QFHResult::collapse_threshold)
        .def_readonly("null_state_count", &sep::quantum::QFHResult::null_state_count)
        .def_readonly("flip_count", &sep::quantum::QFHResult::flip_count)
        .def_readonly("rupture_count", &sep::quantum::QFHResult::rupture_count)
        .def_readonly("flip_ratio", &sep::quantum::QFHResult::flip_ratio)
        .def_readonly("entropy", &sep::quantum::QFHResult::entropy)
        .def_readonly("events", &sep::quantum::QFHResult::events)
        .def_readonly("aggregated_events", &sep::quantum::QFHResult::aggregated_events);

    py::class_<Metrics>(m, "Metrics")
        .def_readonly("coherence", &Metrics::coherence)
        .def_readonly("stability", &Metrics::stability)
        .def_readonly("entropy", &Metrics::entropy)
        .def_readonly("rupture", &Metrics::rupture)
        .def_readonly("lambda_hazard", &Metrics::lambda_hazard)
        .def_readonly("sig_c", &Metrics::sig_c)
        .def_readonly("sig_s", &Metrics::sig_s)
        .def_readonly("sig_e", &Metrics::sig_e);

    m.def("analyze_bits", &analyze_bits_native, "Analyze a window of bits using the native manifold kernel");
    m.def("analyze_window", &analyze_bits_detailed, "Return the full native QFH result for a bit window");
    m.def("transform_rich", &sep::quantum::transform_rich, "Transform bits into QFH events");
    m.def("aggregate_events", &sep::quantum::aggregate, "Aggregate consecutive QFH events");
}
