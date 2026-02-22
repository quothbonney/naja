#pragma once

#include <Eigen/Dense>

namespace naja::rounding {

struct DikinDirections {
    Eigen::MatrixXd V;       // (d, k) — the escape directions in reduced space
    Eigen::MatrixXd Av;      // (m, k) — precomputed A * V for fast chord computation
    int k;                   // number of directions
};

// Compute Dikin escape directions from the barrier Hessian.
//
// When n_base_rows > 0, only uses extra constraint rows (indices >= n_base_rows)
// to target KO/conditioning-induced pinches (delta-Dikin).
// When n_base_rows == 0, uses ALL constraints (full-Hessian Dikin).
//
// The directions are the top-k eigenvectors of:
//   H = sum_{i in selected_rows} (1/delta_i^2) n_i n_i'
// where n_i = a_i / ||a_i|| and delta_i = (b_i - a_i'x_c) / ||a_i||.
//
// eigenvalue_ratio: eigenvalue must exceed this multiple of median to be selected.
//   Use 10.0 for delta-Dikin (sparse, high signal), 1.0 for full-Hessian (always
//   take max_directions).
DikinDirections compute_dikin_directions(
    const Eigen::MatrixXd& A,       // (m, d) full augmented constraints
    const Eigen::VectorXd& b,       // (m,)
    const Eigen::VectorXd& x_c,     // (d,) interior point
    int n_base_rows,                 // number of base constraint rows (0 = use all)
    int max_directions = 32,
    double tight_factor = 50.0,
    double eigenvalue_ratio = 10.0);

} // namespace naja::rounding

