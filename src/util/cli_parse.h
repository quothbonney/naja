#pragma once

#include <stdexcept>
#include <string>

namespace naja::util::cli_parse {

inline void require_int_ge(const std::string& flag, int value, int min_value) {
    if (value < min_value) {
        throw std::invalid_argument("invalid " + flag);
    }
}

inline void require_double_ge(const std::string& flag, double value, double min_value) {
    if (value < min_value) {
        throw std::invalid_argument("invalid " + flag);
    }
}

inline void require_double_in_01(const std::string& flag, double value) {
    if (value < 0.0 || value > 1.0) {
        throw std::invalid_argument("invalid " + flag);
    }
}

} // namespace naja::util::cli_parse


