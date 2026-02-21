#pragma once

#include <Eigen/Dense>

namespace naja::rounding {

struct DikinDirections {
    Eigen::MatrixXd V;       // (d, k) — the escape directions in reduced space
    Eigen::MatrixXd Av;      // (m, k) — precomputed A * V for fast chord computation
    int k;                   // number of directions
};

// Compute Dikin escape directions from the delta barrier Hessian.
//
// Only uses the contribution of extra constraint rows (indices >= n_base_rows)
// to the barrier metric, so the directions target what the KO/conditioning
// introduced, not base rounding imperfections.
//
// The directions are the top-k eigenvectors of:
//   H_delta = sum_{i in extra_rows} (1/delta_i^2) n_i n_i'
// where n_i = a_i / ||a_i|| and delta_i = (b_i - a_i'x_c) / ||a_i||.
//
// Returns at most k directions. If no extra rows are tight, returns k=0.
DikinDirections compute_dikin_directions(
    const Eigen::MatrixXd& A,       // (m, d) full augmented constraints
    const Eigen::VectorXd& b,       // (m,)
    const Eigen::VectorXd& x_c,     // (d,) interior point
    int n_base_rows,                 // number of base constraint rows (extra rows start after this)
    int max_directions = 32,
    double tight_factor = 50.0);

} // namespace naja::rounding

