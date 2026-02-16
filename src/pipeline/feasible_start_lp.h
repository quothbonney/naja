#pragma once

#include <Eigen/Dense>

#include <utility>

namespace naja::pipeline {

// Solve: maximize r
// s.t. A x + r * ||a_i||_1 <= b_i for all rows i, r >= 0.
// Returns (x, r). Throws on solver failure or infeasible LP.
std::pair<Eigen::VectorXd, double> axis_aligned_cube_center_lp_gurobi(const Eigen::MatrixXd& A,
                                                                      const Eigen::VectorXd& b);

} // namespace naja::pipeline


