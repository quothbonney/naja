#include <Eigen/Dense>

#include <cmath>
#include <stdexcept>

#include "pipeline/feasible_start_lp.h"

int main() {
    // Box: -1 <= x_j <= 1 for j=0..2
    // A = [ I ; -I ], b = [1;1;1;1;1;1]
    const int n = 3;
    Eigen::MatrixXd A(2 * n, n);
    A.setZero();
    for (int j = 0; j < n; ++j) {
        A(j, j) = 1.0;
        A(n + j, j) = -1.0;
    }
    Eigen::VectorXd b(2 * n);
    b.setOnes();

    auto [x, r] = naja::pipeline::axis_aligned_cube_center_lp_gurobi(A, b);
    if (x.size() != n) throw std::runtime_error("wrong x dimension");

    const double tol = 1e-7;
    if (std::abs(r - 1.0) > 1e-6) throw std::runtime_error("unexpected r");
    for (int j = 0; j < n; ++j) {
        if (std::abs(x[j]) > tol) throw std::runtime_error("unexpected x");
    }

    Eigen::VectorXd slack = b - A * x;
    if ((slack.array() < -1e-9).any()) throw std::runtime_error("x infeasible");
    if ((slack.array() < r - 1e-6).any()) throw std::runtime_error("cube not inside polytope");

    return 0;
}


