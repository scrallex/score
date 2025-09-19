#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace sep::quantum {

enum class QFHState {
    NULL_STATE,
    STABLE,
    UNSTABLE,
    COLLAPSING,
    COLLAPSED,
    RECOVERING,
    FLIP,
    RUPTURE
};

struct QFHEvent {
    std::uint32_t index{0};
    QFHState state{QFHState::NULL_STATE};
    std::uint8_t bit_prev{0};
    std::uint8_t bit_curr{0};

    bool operator==(const QFHEvent& other) const;
};

struct QFHAggregateEvent {
    std::uint32_t index{0};
    QFHState state{QFHState::NULL_STATE};
    std::uint32_t count{1};
};

struct QFHOptions {
    double coherence_threshold = 0.7;
    double stability_threshold = 0.8;
    double collapse_threshold = 0.5;
    int max_iterations = 1000;
    bool enable_damping = true;
    double damping_factor = 0.95;
    double entropy_weight = 0.30;
    double coherence_weight = 0.20;
};

struct QFHResult {
    double coherence = 0.0;
    double stability = 0.0;
    double confidence = 0.0;
    bool collapse_detected = false;
    double rupture_ratio = 0.0;
    QFHState final_state = QFHState::STABLE;
    std::vector<QFHEvent> events;
    double collapse_threshold = 0.5;
    std::vector<QFHAggregateEvent> aggregated_events;
    std::uint32_t null_state_count = 0;
    std::uint32_t flip_count = 0;
    std::uint32_t rupture_count = 0;
    double flip_ratio = 0.0;
    double entropy = 0.0;
};

namespace bitspace {
struct DampedValue {
    double final_value{0.0};
    double confidence{0.0};
    bool converged{false};
    std::vector<double> path;
    double lambda{0.0};
    std::size_t start_index{0};
};
}

class QFHProcessor {
public:
    QFHProcessor() = default;
    virtual ~QFHProcessor() = default;

    virtual std::optional<QFHState> process(std::uint8_t current_bit);
    virtual void reset();

protected:
    std::optional<std::uint8_t> prev_bit;
};

class QFHBasedProcessor : public QFHProcessor {
public:
    explicit QFHBasedProcessor(const QFHOptions& options);
    ~QFHBasedProcessor() override = default;

    QFHResult analyze(const std::vector<std::uint8_t>& data);
    void reset() override;

    bitspace::DampedValue integrateFutureTrajectories(const std::vector<std::uint8_t>& bitstream, std::size_t current_index);
    double matchKnownPaths(const std::vector<double>& trajectory);
    std::vector<std::uint8_t> convertToBits(const std::vector<std::uint32_t>& data);
    double calculateCosineSimilarity(const std::vector<double>& a, const std::vector<double>& b);
    std::optional<QFHState> detectTransition(std::uint32_t prev_bit, std::uint32_t current_bit);
    bool detectCollapse(const QFHResult& result) const;

private:
    QFHOptions options_;
    QFHState current_state_{QFHState::STABLE};
    std::uint32_t prev_bit_{0};
};

std::vector<QFHEvent> transform_rich(const std::vector<std::uint8_t>& bits);
std::vector<QFHAggregateEvent> aggregate(const std::vector<QFHEvent>& events);

} // namespace sep::quantum
