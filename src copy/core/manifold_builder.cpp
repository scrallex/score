#include "manifold_builder.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <iomanip>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "qfh.h"

namespace sep {

nlohmann::json buildManifold(const std::vector<Candle>& candles, const std::string& instrument_hint) {
    nlohmann::json result;
    const std::string instrument = instrument_hint.empty() ? "UNKNOWN" : instrument_hint;
    result["instrument"] = instrument;

    if (candles.empty()) {
        result["count"] = 0;
        result["signals"] = nlohmann::json::array();
        return result;
    }

    const uint64_t t0 = candles.front().timestamp;
    const uint64_t t1 = candles.back().timestamp;

    // Build enriched bitstream reflecting momentum, volatility, and volume dynamics.
    std::vector<uint8_t> bits;
    bits.reserve(candles.size() > 1 ? (candles.size() - 1) : 0);
    for (size_t i = 1; i < candles.size(); ++i) {
        const auto& prev = candles[i - 1];
        const auto& curr = candles[i];

        const bool price_up = curr.close >= prev.close;
        const bool range_expanding = (curr.high - curr.low) >= (prev.high - prev.low);
        const bool volume_increasing = curr.volume >= prev.volume;

        uint8_t bit_value = price_up ? 1 : 0;
        if (!range_expanding && !volume_increasing) {
            bit_value = 0;  // Quiet regime dampens the signal.
        }

        bits.push_back(bit_value);
    }

    const size_t max_signals = 512;
    nlohmann::json signals = nlohmann::json::array();

    static const double signature_precision = []() {
        if (const char* env = std::getenv("ECHO_SIGNATURE_PRECISION")) {
            try {
                return std::max(0.0, std::min(6.0, std::stod(env)));
            } catch (...) {
                return 2.0;
            }
        }
        return 2.0;
    }();

    const double scale = std::pow(10.0, signature_precision);
    auto bucket = [&](double value) {
        const double clamped = std::clamp(value, 0.0, 1.0);
        return std::round(clamped * scale) / scale;
    };

    auto make_signature = [&](double c, double s, double e) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(static_cast<int>(signature_precision));
        oss << "c" << bucket(c) << "_s" << bucket(s) << "_e" << bucket(e);
        return oss.str();
    };

    static const uint64_t repetition_window_ms = []() -> uint64_t {
        if (const char* env = std::getenv("ECHO_LOOKBACK_MINUTES")) {
            try {
                double minutes = std::stod(env);
                minutes = std::clamp(minutes, 1.0, 1440.0);
                return static_cast<uint64_t>(minutes * 60.0 * 1000.0);
            } catch (...) {
                return static_cast<uint64_t>(60ULL * 60ULL * 1000ULL);
            }
        }
        return static_cast<uint64_t>(60ULL * 60ULL * 1000ULL);
    }();

    std::unordered_map<std::string, std::deque<uint64_t>> repetition_history;

    if (!bits.empty()) {
        sep::quantum::QFHOptions opts{};
        sep::quantum::QFHBasedProcessor proc(opts);

        size_t window = bits.size();
        if (window > 128) {
            window = 128;
        }
        if (window >= bits.size()) {
            if (bits.size() > 24) {
                window = bits.size() - std::min<size_t>(8, bits.size() / 4);
            } else if (bits.size() > 12) {
                window = 12;
            } else if (bits.size() > 8) {
                window = 9;
            } else {
                window = bits.size();
            }
        }

        if (window >= 8 && bits.size() >= 2) {
            size_t start_i = window;
            if (bits.size() - window > max_signals) {
                start_i = bits.size() - max_signals;
            }

            const size_t step = std::max<size_t>(1, window / 32);
            for (size_t i = start_i; i <= bits.size(); i += step) {
                const size_t begin = i - window;
                if (begin >= bits.size()) break;

                std::vector<uint8_t> sub(bits.begin() + begin, bits.begin() + i);
                const sep::quantum::QFHResult r = proc.analyze(sub);

                const uint64_t ts_ms = candles[begin + 1].timestamp;
                const double price = candles[begin + 1].close;
                const double coherence = static_cast<double>(r.coherence);
                const double stability = 1.0 - static_cast<double>(r.rupture_ratio);
                const double entropy = static_cast<double>(r.entropy);
                const double rupture = static_cast<double>(r.rupture_ratio);

                const std::string signature = make_signature(
                    std::clamp(coherence, 0.0, 1.0),
                    std::clamp(stability, 0.0, 1.0),
                    std::clamp(entropy, 0.0, 1.0)
                );
                auto& history = repetition_history[signature];
                while (!history.empty() && (ts_ms > history.front()) && (ts_ms - history.front() > repetition_window_ms)) {
                    history.pop_front();
                }
                history.push_back(ts_ms);
                const uint64_t first_seen = history.front();
                const uint64_t count = history.size();
                const double hazard_lambda = std::clamp(rupture, 0.0, 1.0);

                nlohmann::json signal = {
                    {"timestamp_ns", ts_ms * 1000000ULL},
                    {"price", price},
                    {"state", r.collapse_detected ? "collapsed" : "live"},
                    {"metrics", {
                        {"coherence", coherence},
                        {"stability", stability},
                        {"entropy", entropy},
                        {"rupture", rupture}
                    }},
                    {"coeffs", {
                        {"lambda_hazard", hazard_lambda}
                    }},
                    {"repetition", {
                        {"signature", signature},
                        {"count_1h", static_cast<uint64_t>(count)},
                        {"first_seen_ms", first_seen}
                    }},
                    {"coherence", coherence},
                    {"stability", stability},
                    {"entropy", entropy},
                    {"rupture", rupture},
                    {"lambda_hazard", hazard_lambda}
                };
                signals.push_back(std::move(signal));
            }
        }
    }
    result["count"] = static_cast<uint64_t>(candles.size());
    result["t0_ms"] = t0;
    result["t1_ms"] = t1;
    result["signals"] = std::move(signals);

    try {
        double coh = 0.0;
        double stab = 0.0;
        double ent = 0.0;
        double rup = 0.0;
        if (result.contains("signals") && result["signals"].is_array() && !result["signals"].empty()) {
            const auto& s = result["signals"].back();
            if (s.contains("metrics")) {
                const auto& m = s["metrics"];
                coh = m.value("coherence", s.value("coherence", 0.0));
                stab = m.value("stability", s.value("stability", 0.0));
                ent = m.value("entropy", s.value("entropy", 0.0));
                rup = m.value("rupture", s.value("rupture", 0.0));
            } else {
                coh = s.value("coherence", 0.0);
                stab = s.value("stability", 0.0);
                ent = s.value("entropy", 0.0);
                rup = s.value("rupture", 0.0);
            }
        }
        result["metrics"] = {
            {"coherence", coh},
            {"stability", stab},
            {"entropy", ent},
            {"rupture", rup}
        };
    } catch (...) {
        // Leave metrics absent on failure – downstream consumers fall back gracefully.
    }

    try {
        double sigma_eff = 0.0;
        if (candles.size() >= 3) {
            std::vector<double> rets;
            rets.reserve(candles.size() - 1);
            for (size_t i = 1; i < candles.size(); ++i) {
                const double c1 = candles[i - 1].close;
                const double c2 = candles[i].close;
                if (c1 > 0.0 && c2 > 0.0) {
                    rets.push_back(std::log(c2 / c1));
                }
            }
            if (rets.size() >= 2) {
                double mean = 0.0;
                for (double x : rets) mean += x;
                mean /= static_cast<double>(rets.size());

                double var = 0.0;
                for (double x : rets) {
                    const double d = x - mean;
                    var += d * d;
                }
                var /= static_cast<double>(rets.size() - 1);
                sigma_eff = std::sqrt(std::max(0.0, var));
            }
        }

        double lambda_pmin = 0.0;
        double t_sum_sec = 0.0;
        double r_sum = 0.0;
        if (result.contains("signals") && result["signals"].is_array() && result["signals"].size() >= 2) {
            const auto& arr = result["signals"];
            for (size_t i = 1; i < arr.size(); ++i) {
                double r = 0.0;
                if (arr[i].contains("metrics")) {
                    r = arr[i]["metrics"].value("rupture", arr[i].value("rupture", 0.0));
                } else {
                    r = arr[i].value("rupture", 0.0);
                }
                const double t_i = static_cast<double>(arr[i].value("timestamp_ns", 0ULL)) / 1e9;
                const double t_j = static_cast<double>(arr[i - 1].value("timestamp_ns", 0ULL)) / 1e9;
                const double dt = std::max(1e-6, t_i - t_j);
                r_sum += r;
                t_sum_sec += dt;
            }
            if (t_sum_sec > 0.0) {
                const double lambda_per_sec = r_sum / t_sum_sec;
                lambda_pmin = lambda_per_sec * 60.0;
            }
        }

        const double lambda_prob = 1.0 - std::exp(-std::max(0.0, lambda_pmin));
        result["coeffs"] = {
            {"sigma_eff", sigma_eff},
            {"lambda", std::max(0.0, std::min(1.0, lambda_prob))}
        };
    } catch (...) {
        // Leave coeffs absent on failure.
    }

    return result;
}

}  // namespace sep
