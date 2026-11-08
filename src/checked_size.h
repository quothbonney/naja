#pragma once

#include <cstddef>
#include <cstdint>
#include <ios>
#include <limits>
#include <stdexcept>
#include <string>

namespace naja {

inline std::size_t checked_size_mul(std::size_t a, std::size_t b, const char* what) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error(std::string(what) + ": size overflow");
    return a * b;
}

inline int checked_int_dimension(std::int64_t value, const char* what) {
    if (value < 0) throw std::invalid_argument(std::string(what) + ": negative dimension");
    if (value > std::numeric_limits<int>::max())
        throw std::overflow_error(std::string(what) + ": exceeds cuBLAS/int dimension limit; use smaller streamed chunks");
    return static_cast<int>(value);
}

inline int checked_int_product(int a, int b, const char* what) {
    if (a < 0 || b < 0) throw std::invalid_argument(std::string(what) + ": negative dimension");
    return checked_int_dimension(static_cast<std::int64_t>(a) * static_cast<std::int64_t>(b), what);
}

template <typename T>
inline std::size_t checked_matrix_bytes(std::int64_t rows, std::int64_t cols, const char* what) {
    if (rows < 0 || cols < 0) throw std::invalid_argument(std::string(what) + ": negative dimension");
    return checked_size_mul(checked_size_mul(static_cast<std::size_t>(rows),
        static_cast<std::size_t>(cols), what), sizeof(T), what);
}

inline std::streamsize checked_streamsize(std::size_t bytes, const char* what) {
    if (bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
        throw std::overflow_error(std::string(what) + ": exceeds stream size limit");
    return static_cast<std::streamsize>(bytes);
}

} // namespace naja
