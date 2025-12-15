#pragma once

#include <iostream>
#include <string>

namespace naja::status {

// Minimal, GNU-ish phase logging. No buffering, no magic.
inline void phase(bool enabled, const std::string& msg) {
    if (!enabled) return;
    std::cout << "> " << msg << std::endl;
}

inline void kv(bool enabled, const std::string& k, const std::string& v) {
    if (!enabled) return;
    std::cout << k << " :: " << v << std::endl;
}

} // namespace naja::status


