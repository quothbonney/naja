#include "engine/constraint_utils.h"

#include <Eigen/Dense>
#include <iostream>

// Synthetic reproduction of the "point polytope" failure mode:
// Base constraints make x fixed but y free (rank 1 tight set in d=2).
// Extra constraints make y fixed too (rank 2 tight set -> point).
// Relaxing b_extra by eps makes extra constraints non-tight at the start -> rank drops back to 1.

int main() {
    const double tol = 1e-9;
    const double eps = 1e-6;

    // Variables: [x, y]
    Eigen::VectorXd x0(2);
    x0 << 0.0, 0.0;

    // Base constraints: x <= 0, -x <= 0  (forces x==0 at x0, but doesn't constrain y)
    Eigen::MatrixXd A_base(2, 2);
    A_base <<  1.0, 0.0,
              -1.0, 0.0;
    Eigen::VectorXd b_base(2);
    b_base << 0.0, 0.0;

    // Extra constraints: y <= 0, -y <= 0  (forces y==0 at x0)
    Eigen::MatrixXd A_extra(2, 2);
    A_extra << 0.0,  1.0,
               0.0, -1.0;
    Eigen::VectorXd b_extra(2);
    b_extra << 0.0, 0.0;

    Eigen::MatrixXd A_all(4, 2);
    A_all.topRows(2) = A_base;
    A_all.bottomRows(2) = A_extra;
    Eigen::VectorXd b_all(4);
    b_all.head(2) = b_base;
    b_all.tail(2) = b_extra;

    const int r_all = naja::engine::tight_constraint_rank(A_all, b_all, x0, tol);
    if (r_all != 2) {
        std::cerr << "expected tight rank 2 (point), got " << r_all << "\n";
        return 1;
    }

    // Relax only extra b: b_extra += eps
    Eigen::VectorXd b_relaxed = b_all;
    b_relaxed.tail(2).array() += eps;
    const int r_relaxed = naja::engine::tight_constraint_rank(A_all, b_relaxed, x0, tol);
    if (r_relaxed != 1) {
        std::cerr << "expected tight rank 1 after relaxing extra constraints, got " << r_relaxed << "\n";
        return 1;
    }

    std::cout << "ok\n";
    return 0;
}



