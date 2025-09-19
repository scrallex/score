#include "qfh_native.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <execution>
#include <numeric>
#include <vector>

namespace sep::quantum {

bool QFHEvent::operator==(const QFHEvent& other) const {
    return index == other.index && state == other.state &&
           bit_prev == other.bit_prev && bit_curr == other.bit_curr;
}

std::vector<QFHEvent> transform_rich(const std::vector<std::uint8_t>& bits) {
    if (bits.size() < 2) {
        return {};
    }
    const std::size_t n = bits.size() - 1;
    std::vector<QFHEvent> events(n);
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::atomic<bool> invalid{false};
    std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [&](std::size_t i) {
        std::uint8_t prev = bits[i];
        std::uint8_t curr = bits[i + 1];
        if ((prev > 1) || (curr > 1)) {
            invalid.store(true, std::memory_order_relaxed);
            return;
        }
        QFHState state = QFHState::NULL_STATE;
        if ((prev == 0 && curr == 1) || (prev == 1 && curr == 0)) {
            state = QFHState::FLIP;
        } else if (prev == 1 && curr == 1) {
            state = QFHState::RUPTURE;
        }
        events[i] = QFHEvent{static_cast<std::uint32_t>(i), state, prev, curr};
    });
    if (invalid.load(std::memory_order_relaxed)) {
        return {};
    }
    return events;
}

std::vector<QFHAggregateEvent> aggregate(const std::vector<QFHEvent>& events) {
    if (events.empty()) {
        return {};
    }
    std::vector<QFHAggregateEvent> aggregated;
    aggregated.reserve(events.size());
    aggregated.push_back({events.front().index, events.front().state, 1});
    for (std::size_t i = 1; i < events.size(); ++i) {
        if (events[i].state == aggregated.back().state) {
            aggregated.back().count++;
        } else {
            aggregated.push_back({events[i].index, events[i].state, 1});
        }
    }
    return aggregated;
}

std::optional<QFHState> QFHProcessor::process(std::uint8_t current_bit) {
    if (current_bit > 1) {
        return std::nullopt;
    }
    if (!prev_bit.has_value()) {
        prev_bit = current_bit;
        return std::nullopt;
    }
    std::uint8_t prev = *prev_bit;
    std::optional<QFHState> event_state;
    if ((prev == 0 && current_bit == 1) || (prev == 1 && current_bit == 0)) {
        event_state = QFHState::FLIP;
    } else if (prev == 1 && current_bit == 1) {
        event_state = QFHState::RUPTURE;
    } else {
        event_state = QFHState::NULL_STATE;
    }
    prev_bit = current_bit;
    return event_state;
}

void QFHProcessor::reset() {
    prev_bit.reset();
}

QFHBasedProcessor::QFHBasedProcessor(const QFHOptions& options)
    : options_(options) {}

bitspace::DampedValue QFHBasedProcessor::integrateFutureTrajectories(const std::vector<std::uint8_t>& bitstream, std::size_t current_index) {
    bitspace::DampedValue dv;
    if (current_index >= bitstream.size()) {
        return dv;
    }
    const std::size_t window_size = std::min<std::size_t>(20, bitstream.size() - current_index);
    std::vector<std::uint8_t> local_window(bitstream.begin() + current_index, bitstream.begin() + current_index + window_size);
    auto local_events = transform_rich(local_window);
    double local_entropy = 0.5;
    double local_coherence = 0.5;
    if (!local_events.empty()) {
        std::uint32_t null_count = 0;
        std::uint32_t flip_count = 0;
        std::uint32_t rupture_count = 0;
        for (const auto& event : local_events) {
            switch (event.state) {
                case QFHState::NULL_STATE:
                    ++null_count;
                    break;
                case QFHState::FLIP:
                    ++flip_count;
                    break;
                case QFHState::RUPTURE:
                    ++rupture_count;
                    break;
                default:
                    break;
            }
        }
        const float total = static_cast<float>(local_events.size());
        const auto safe_log2 = [](float x) -> float { return (x > 0.0f) ? std::log2(x) : 0.0f; };
        float null_ratio = null_count / total;
        float flip_ratio = flip_count / total;
        float rupture_ratio = rupture_count / total;
        local_entropy = -(null_ratio * safe_log2(null_ratio) + flip_ratio * safe_log2(flip_ratio) + rupture_ratio * safe_log2(rupture_ratio));
        local_entropy = std::fmax(0.05f, std::fmin(1.0f, local_entropy / 1.585f));
        local_coherence = 1.0f - local_entropy;
    }
    double lambda = options_.entropy_weight * local_entropy + options_.coherence_weight * (1.0 - local_coherence);
    lambda = std::fmax(0.01, std::fmin(1.0, lambda));
    dv.lambda = lambda;
    dv.start_index = current_index;
    double accumulated_value = 0.0;
    double current_bit = static_cast<double>(bitstream[current_index]);
    dv.path.clear();
    dv.path.reserve(bitstream.size() - current_index);
    dv.path.push_back(current_bit);
    for (std::size_t j = current_index + 1; j < bitstream.size(); ++j) {
        double future_bit = static_cast<double>(bitstream[j]);
        double delta = std::exp(-lambda * static_cast<double>(j - current_index));
        accumulated_value += (future_bit - current_bit) * delta;
        dv.path.push_back(accumulated_value);
    }
    dv.final_value = accumulated_value;
    if (dv.path.size() > 2) {
        double mean = 0.0;
        for (double v : dv.path) {
            mean += v;
        }
        mean /= dv.path.size();
        double variance = 0.0;
        for (double v : dv.path) {
            variance += (v - mean) * (v - mean);
        }
        variance /= dv.path.size();
        double stability_score = 1.0 / (1.0 + variance);
        dv.confidence = std::fmax(0.0, std::fmin(1.0, stability_score));
        dv.converged = dv.confidence > 0.7;
    } else {
        dv.confidence = 0.5;
    }
    return dv;
}

double QFHBasedProcessor::matchKnownPaths(const std::vector<double>& current_path) {
    if (current_path.size() < 3) {
        return 0.5;
    }
    // Simple heuristic: favour paths with low variance
    double mean = 0.0;
    for (double v : current_path) {
        mean += v;
    }
    mean /= current_path.size();
    double variance = 0.0;
    for (double v : current_path) {
        variance += (v - mean) * (v - mean);
    }
    variance /= current_path.size();
    double score = 1.0 / (1.0 + variance);
    return std::fmax(0.0, std::fmin(1.0, score));
}

std::vector<std::uint8_t> QFHBasedProcessor::convertToBits(const std::vector<std::uint32_t>& values) {
    std::vector<std::uint8_t> bits;
    bits.reserve(values.size() * 32);
    for (std::uint32_t value : values) {
        for (int i = 0; i < 32; ++i) {
            bits.push_back((value >> i) & 1u);
        }
    }
    return bits;
}

double QFHBasedProcessor::calculateCosineSimilarity(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size() || a.empty()) {
        return 0.0;
    }
    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);
    if (norm_a == 0.0 || norm_b == 0.0) {
        return 0.0;
    }
    return dot / (norm_a * norm_b);
}

std::optional<QFHState> QFHBasedProcessor::detectTransition(std::uint32_t prev_bit, std::uint32_t current_bit) {
    if ((prev_bit == 0 && current_bit == 1) || (prev_bit == 1 && current_bit == 0)) {
        return QFHState::FLIP;
    }
    if (prev_bit == 1 && current_bit == 1) {
        return QFHState::RUPTURE;
    }
    return QFHState::NULL_STATE;
}

bool QFHBasedProcessor::detectCollapse(const QFHResult& result) const {
    return result.rupture_ratio >= options_.collapse_threshold;
}

QFHResult QFHBasedProcessor::analyze(const std::vector<std::uint8_t>& bits) {
    QFHResult result;
    result.collapse_threshold = options_.collapse_threshold;
    result.events = transform_rich(bits);
    result.aggregated_events = aggregate(result.events);
    for (const auto& event : result.events) {
        switch (event.state) {
            case QFHState::NULL_STATE:
                ++result.null_state_count;
                break;
            case QFHState::FLIP:
                ++result.flip_count;
                break;
            case QFHState::RUPTURE:
                ++result.rupture_count;
                break;
            default:
                break;
        }
    }
    if (!result.events.empty()) {
        result.rupture_ratio = static_cast<double>(result.rupture_count) / static_cast<double>(result.events.size());
        result.flip_ratio = static_cast<double>(result.flip_count) / static_cast<double>(result.events.size());
    }
    if (!result.events.empty()) {
        float null_ratio = static_cast<float>(result.null_state_count) / static_cast<float>(result.events.size());
        float flip_ratio = static_cast<float>(result.flip_ratio);
        float rupture_ratio = static_cast<float>(result.rupture_ratio);
        auto safe_log2 = [](float x) -> float { return (x > 0.0f) ? std::log2(x) : 0.0f; };
        result.entropy = -(null_ratio * safe_log2(null_ratio) + flip_ratio * safe_log2(flip_ratio) + rupture_ratio * safe_log2(rupture_ratio));
        result.entropy = std::fmax(0.05f, std::fmin(1.0f, result.entropy / 1.585f));
        float pattern_coherence = 1.0f - result.entropy;
        float stability_factor = 1.0f - static_cast<float>(result.rupture_ratio);
        float consistency_factor = 1.0f - static_cast<float>(result.flip_ratio);
        result.coherence = pattern_coherence * 0.6f + stability_factor * 0.3f + consistency_factor * 0.1f;
        result.coherence = std::fmax(0.01f, std::fmin(0.99f, result.coherence));
    }
    if (!bits.empty() && bits.size() > 10) {
        bitspace::DampedValue dv = integrateFutureTrajectories(bits, 0);
        double trajectory_confidence = matchKnownPaths(dv.path);
        float pattern_coherence = static_cast<float>(result.coherence);
        float trajectory_coherence = static_cast<float>(trajectory_confidence);
        result.coherence = 0.3f * trajectory_coherence + 0.7f * pattern_coherence;
        if (std::abs(dv.final_value) < 2.0) {
            float stability_factor = 1.0f / (1.0f + 0.1f * std::abs(static_cast<float>(dv.final_value)));
            result.coherence *= stability_factor;
        }
        result.confidence = dv.confidence;
    }
    result.coherence = std::fmax(0.0f, std::fmin(1.0f, result.coherence));
    result.stability = 1.0 - result.rupture_ratio;
    result.collapse_detected = detectCollapse(result);
    return result;
}

void QFHBasedProcessor::reset() {
    QFHProcessor::reset();
    current_state_ = QFHState::STABLE;
    prev_bit_ = 0;
}

} // namespace sep::quantum
