#pragma once

#include <Eigen/Dense>

namespace naja::rounding {

struct BarrierRotation {
    Eigen::MatrixXd Q;           // (d, d) orthogonal eigenvectors of H
    Eigen::VectorXd eigenvalues; // (d,) eigenvalues, ascending
};

// Eigen-decompose the log-barrier Hessian
//   H(x_c) = A' diag(1/s^2) A,    s = b - A x_c
// and return the orthogonal Q that diagonalizes it.
//
// Applying x = Q y rotates coordinate CHR into the eigenbasis of
// the barrier metric: every coordinate step then moves along a
// principal direction of the facet-induced coupling structure.
//
// Since Q is orthogonal, chord lengths are preserved.
//
// Guardrails:
//   slack clamping:  s_i <- max(s_i, tau)         (one tight facet cannot dominate)
//   ridge:           H <- H + lam*(tr H / d) * I  (numerical stability)
BarrierRotation compute_barrier_rotation(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& x_c,
    bool verbose = false);

} // namespace naja::rounding

