#include <Eigen/Dense>

#include <stdexcept>

#include "engine/start_feasibility.h"

static void expect_throws_infeasible() {
    Eigen::MatrixXd A(1, 1);
    Eigen::VectorXd b(1);
    Eigen::VectorXd x0(1);
    A(0, 0) = 1.0;
    b(0) = 0.0;
    x0(0) = 2.0;
    bool threw = false;
    try {
        naja::engine::require_feasible_start(A, b, x0, 1e-9, "unit");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    if (!threw) {
        throw std::runtime_error("expected infeasible start to throw");
    }
}

static void expect_no_throw_feasible() {
    Eigen::MatrixXd A(2, 2);
    Eigen::VectorXd b(2);
    Eigen::VectorXd x0(2);
    // Constraints:
    // x0 <= 1
    // -x1 <= 0  (x1 >= 0)
    A << 1.0, 0.0,
         0.0, -1.0;
    b << 1.0, 0.0;
    x0 << 1.0, 0.0;
    naja::engine::require_feasible_start(A, b, x0, 1e-9, "unit");
}

int main() {
    expect_throws_infeasible();
    expect_no_throw_feasible();
    return 0;
}


