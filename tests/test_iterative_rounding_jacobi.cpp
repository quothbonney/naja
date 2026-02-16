#include "engine/iterative_rounding.h"

#include <cmath>
#include <iostream>

// Verify the Jacobi rotation returned by jacobi_rotation_cs diagonalizes a symmetric 2x2 covariance.
static bool check_diag(double a, double b, double c) {
    auto [cs, sn] = naja::engine::jacobi_rotation_cs(a, b, c);
    // Closed-form off-diagonal of R^T S R for S=[a c; c b], R=[cs -sn; sn cs]:
    // d01 = (b-a) cs sn + c (cs^2 - sn^2)
    const double d01 = (b - a) * cs * sn + c * (cs * cs - sn * sn);
    const double d10 = d01;

    if (!(std::abs(d01) < 1e-10 && std::abs(d10) < 1e-10)) {
        std::cerr << "expected off-diagonal ~0; got d01=" << d01 << " d10=" << d10
                  << " for a=" << a << " b=" << b << " c=" << c << "\n";
        return false;
    }
    return true;
}

int main() {
    // Generic correlated case
    if (!check_diag(4.0, 1.0, 1.2)) return 1;
    // Another correlated case
    if (!check_diag(0.5, 3.0, -0.7)) return 1;
    // Already diagonal: should keep near-diagonal
    if (!check_diag(2.0, 5.0, 0.0)) return 1;

    std::cout << "ok\n";
    return 0;
}


