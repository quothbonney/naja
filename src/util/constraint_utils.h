#pragma once

#include <Eigen/Dense>

namespace naja::util {

// Compute rank of the matrix of constraints that are tight at x, i.e. those rows i with (b_i - A_i x) <= tol.
int tight_constraint_rank(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const Eigen::VectorXd& x, double tol);

} // namespace naja::util


