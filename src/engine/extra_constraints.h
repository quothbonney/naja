#pragma once

#include <Eigen/Dense>
#include <string>

namespace naja::engine {

enum class ExtraConstraintsMode {
    Auto,
    Ignore,
    Require,
};

ExtraConstraintsMode parse_extra_constraints_mode(const std::string& s);

// Augment (A,b) with (A_extra,b_extra) based on mode.
// If extra_eps > 0, relaxes extra constraints: b_extra += extra_eps.
// Returns true if extra constraints were actually loaded and appended.
bool maybe_augment_extra_constraints(Eigen::MatrixXd& A,
                                    Eigen::VectorXd& b,
                                    const Eigen::MatrixXd* A_extra,
                                    const Eigen::VectorXd* b_extra,
                                    ExtraConstraintsMode mode,
                                    double extra_eps);

} // namespace naja::engine



