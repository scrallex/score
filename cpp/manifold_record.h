#pragma once

#include <cstdint>

#pragma pack(push, 1)
struct ManifoldRecord {
    std::uint64_t file_id;
    std::uint64_t window_index;
    std::uint64_t byte_start;
    std::uint32_t window_bytes;
    std::uint32_t stride_bytes;
    float coherence;
    float stability;
    float entropy;
    float rupture;
    float lambda_hazard;
    std::uint16_t sig_c;
    std::uint16_t sig_s;
    std::uint16_t sig_e;
    std::uint16_t reserved;
    std::uint32_t flags;
};
#pragma pack(pop)

static_assert(sizeof(ManifoldRecord) == 64, "ManifoldRecord must be 64 bytes");
