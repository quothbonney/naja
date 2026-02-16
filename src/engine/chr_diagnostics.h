#pragma once

#include <Eigen/Dense>

namespace naja::engine {

struct ChrAxisChordSummary {
    // Per-coordinate chord length at x0 for CHR axis moves, using the same parameterization as the GPU kernel:
    // inv_dist = a_{r,i} / slack_r, chord = (1/max(inv_dist)) - (1/min(inv_dist)).
    Eigen::VectorXd chord_len;
};

// Compute per-axis chord lengths at x0 for constraints A x <= b.
// This is a rounding quality diagnostic for coordinate hit-and-run.
ChrAxisChordSummary chr_axis_chords(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const Eigen::VectorXd& x0);

} // namespace naja::engine



