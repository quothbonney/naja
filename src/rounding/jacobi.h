#pragma once

#include <utility>

namespace naja::rounding {

// Given the 2x2 covariance block for coordinates (i,j):
//   [ var_i   cov_ij ]
//   [ cov_ij  var_j  ]
// return (c,s) where theta = 0.5 * atan2(2*cov_ij, var_j - var_i),
// and c = cos(theta), s = sin(theta).
//
// This is the classic Jacobi rotation angle that diagonalizes a symmetric 2x2 matrix.
std::pair<double, double> jacobi_rotation_cs(double var_i, double var_j, double cov_ij);

} // namespace naja::rounding


