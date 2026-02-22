#pragma once

#include <Eigen/Dense>
#include "rounding/schedule_io.h"

namespace naja::rounding {

// Compute a Jacobi pair schedule directly from the log-barrier Hessian
//   H(x_c) = A' diag(1/delta^2) A
// where delta_r = (b_r - a_r' x_c) / ||a_r|| is the Euclidean distance to facet r.
//
// Row normalization is built in: we weight by 1/delta^2, not 1/slack^2, so the
// result is invariant to constraint scaling.
//
// Pairs are chosen by greedy matching on the barrier-metric correlation:
//   rho_ij = |G_ij| / sqrt(G_ii G_jj),  G = B' B,  B = diag(1/delta) * A_norm.
//
// Then for each pair (i,j), the Jacobi angle that diagonalizes the 2x2 block
// of G is computed via the standard atan2 formula.
//
// This replaces warmup-based Jacobi: the geometry is known analytically from the
// constraints, so no chain traversal is needed to discover the slow modes.
PairSchedule compute_barrier_pair_schedule(
    const Eigen::MatrixXd& A,       // (m, d) constraint matrix
    const Eigen::VectorXd& b,       // (m,) rhs
    const Eigen::VectorXd& x_c,     // (d,) interior point
    bool verbose = false);

} // namespace naja::rounding

