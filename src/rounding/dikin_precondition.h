#pragma once

#include <Eigen/Dense>

namespace naja::rounding {

struct DikinResult {
    Eigen::MatrixXd P;          // corrective transform (d x d): y_new = P * (y - x_c) maps to preconditioned space
    Eigen::MatrixXd A_new;      // transformed constraints: A_new * y_new <= b_new
    Eigen::VectorXd b_new;
    Eigen::VectorXd x0_new;     // start point in new coords (should be near origin)
    int n_corrected;            // number of directions that were rescaled
};

// Dikin-ellipsoid-based analytic preconditioner for augmented polytopes.
//
// Given A y <= b with interior point x_c:
//   1. Compute normalized distances to each facet: delta_i = (b_i - a_i'x_c) / ||a_i||
//   2. Build barrier Hessian from the tightest constraints:
//      H = sum_{i in T} (1/delta_i^2) n_i n_i'   where n_i = a_i / ||a_i||
//   3. Eigen-decompose H, take top-k eigenvectors (the pinched directions)
//   4. Compute actual chord lengths along those directions
//   5. Build low-rank corrective transform P = I + V_k (diag(alpha) - I) V_k'
//      where alpha_j = clip(target_chord / chord_j, alpha_min, alpha_max)
//   6. Transform the constraint system: A_new = A * P^{-1}, b_new = b - A * x_c
//
// This is cheap (one eigendecomposition of a d x d matrix + k chord computations)
// and targets exactly the constraint-induced pinches that break CHR mixing.
//
// Parameters:
//   k_directions:  max number of pinched directions to correct (default 64)
//   tight_factor:  include constraints with delta_i <= tight_factor * min(delta) (default 100)
//   target_chord:  desired chord length after correction (0 = use median of uncorrected chords)
//   alpha_max:     max rescaling factor per direction (default 1000, prevents blowup)
DikinResult dikin_precondition(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& x_c,
    int k_directions = 64,
    double tight_factor = 100.0,
    double target_chord = 0.0,
    double alpha_max = 1000.0);

} // namespace naja::rounding

