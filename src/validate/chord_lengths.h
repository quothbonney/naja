#pragma once

#include <Eigen/Dense>

namespace naja::validate {

struct ChordSummary {
    Eigen::VectorXd chord_len;  // per-coordinate chord length at x0
};

// Compute per-axis chord lengths at x0 for constraints A x <= b.
// Diagnostic for rounding quality: healthy polytopes have chord_len >> 0 in all dims.
ChordSummary chr_axis_chords(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const Eigen::VectorXd& x0);

} // namespace naja::validate

