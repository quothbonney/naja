#pragma once

#include <stdexcept>
#include <string>

#include "runtime_config.h"

namespace naja::rounding {

struct RoundingConfig {
    double pair_prob = 0.0;
    int resync_interval = 0;
    int iter_rounding_passes = 0;
    int iter_rounding_warmup = 0;
    std::string pair_schedule_path;
};

inline RoundingConfig from_runtime_config(const RuntimeConfig& cfg) {
    RoundingConfig out;
    out.pair_prob = cfg.PAIR_PROB;
    out.resync_interval = cfg.RESYNC_INTERVAL;
    out.iter_rounding_passes = cfg.ITER_ROUNDING_PASSES;
    out.iter_rounding_warmup = cfg.ITER_ROUNDING_WARMUP;
    out.pair_schedule_path = cfg.PAIR_SCHEDULE;
    return out;
}

inline void validate_rounding_config(const RoundingConfig& cfg) {
    if (cfg.pair_prob < 0.0 || cfg.pair_prob > 1.0) {
        throw std::invalid_argument("PAIR_PROB must be in [0,1]");
    }
    if (cfg.resync_interval < 0) {
        throw std::invalid_argument("RESYNC_INTERVAL must be >= 0");
    }
    if (cfg.iter_rounding_passes < 0) {
        throw std::invalid_argument("ITER_ROUNDING_PASSES must be >= 0");
    }
    if (cfg.iter_rounding_warmup < 0) {
        throw std::invalid_argument("ITER_ROUNDING_WARMUP must be >= 0");
    }
}

} // namespace naja::rounding


