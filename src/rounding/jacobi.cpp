#include "rounding/jacobi.h"

#include <cmath>
#include <stdexcept>

namespace naja::rounding {

std::pair<double, double> jacobi_rotation_cs(double var_i, double var_j, double cov_ij) {
    if (!(var_i >= 0.0) || !(var_j >= 0.0)) {
        throw std::invalid_argument("jacobi_rotation_cs: variances must be nonnegative");
    }
    if (!std::isfinite(var_i) || !std::isfinite(var_j) || !std::isfinite(cov_ij)) {
        throw std::invalid_argument("jacobi_rotation_cs: inputs must be finite");
    }

    if (cov_ij == 0.0) {
        return {1.0, 0.0};
    }

    const double theta = 0.5 * std::atan2(2.0 * cov_ij, var_i - var_j);
    return {std::cos(theta), std::sin(theta)};
}

} // namespace naja::rounding


