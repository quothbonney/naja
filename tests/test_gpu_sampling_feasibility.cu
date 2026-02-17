// Hard correctness test: sample several polytopes on GPU and verify that
// EVERY SINGLE output sample satisfies A*x <= b within tolerance.
// Polytopes tested:
//   1. 10D hypercube [-1,1]^10 (20 constraints)
//   2. 8D simplex {x>=0, sum(x)<=1} (9 constraints)
//   3. 6D thin slab: cube + |sum(x)| <= eps (14 constraints, near-degenerate)
//   4. 15D random dense polytope with 60 constraints

#include <Eigen/Dense>
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

#include "dmatrix.h"
#include "dvector.h"
#include "gpusamplers.h"
#include "device_utils.h"

static int check_feasibility(const Eigen::MatrixXd& A,
                             const Eigen::VectorXd& b,
                             const Eigen::MatrixXd& samples,
                             double tol,
                             const std::string& label) {
    const int m = A.rows();
    const int n_samples = samples.cols();
    int violations = 0;
    double worst = 0.0;
    int worst_row = -1;
    int worst_col = -1;

    for (int j = 0; j < n_samples; ++j) {
        Eigen::VectorXd residual = A * samples.col(j) - b;
        for (int i = 0; i < m; ++i) {
            if (residual(i) > tol) {
                ++violations;
                if (residual(i) > worst) {
                    worst = residual(i);
                    worst_row = i;
                    worst_col = j;
                }
            }
        }
    }

    if (violations > 0) {
        std::cerr << "[FAIL] " << label << ": " << violations
                  << " constraint violations out of " << (m * n_samples)
                  << " checks. worst=" << worst
                  << " at row=" << worst_row << " sample=" << worst_col << "\n";
    } else {
        std::cout << "[PASS] " << label << ": " << n_samples
                  << " samples, all feasible (tol=" << tol << ")\n";
    }
    return violations;
}

// Build [-1,1]^d hypercube: x_i <= 1, -x_i <= 1
static void make_cube(int d, Eigen::MatrixXd& A, Eigen::VectorXd& b, Eigen::VectorXd& x0) {
    A = Eigen::MatrixXd::Zero(2 * d, d);
    b = Eigen::VectorXd::Ones(2 * d);
    for (int i = 0; i < d; ++i) {
        A(2 * i, i) = 1.0;
        A(2 * i + 1, i) = -1.0;
    }
    x0 = Eigen::VectorXd::Zero(d);
}

// Build standard simplex in R^d: -x_i <= 0, sum(x_i) <= 1
static void make_simplex(int d, Eigen::MatrixXd& A, Eigen::VectorXd& b, Eigen::VectorXd& x0) {
    A = Eigen::MatrixXd::Zero(d + 1, d);
    b = Eigen::VectorXd::Zero(d + 1);
    for (int i = 0; i < d; ++i) {
        A(i, i) = -1.0; // -x_i <= 0
    }
    A.row(d).setOnes(); // sum(x) <= 1
    b(d) = 1.0;
    x0 = Eigen::VectorXd::Constant(d, 1.0 / (d + 1));
}

// Build thin slab: [-1,1]^d + |sum(x)| <= eps
static void make_thin_slab(int d, double eps, Eigen::MatrixXd& A, Eigen::VectorXd& b, Eigen::VectorXd& x0) {
    A = Eigen::MatrixXd::Zero(2 * d + 2, d);
    b = Eigen::VectorXd::Zero(2 * d + 2);
    for (int i = 0; i < d; ++i) {
        A(2 * i, i) = 1.0;
        A(2 * i + 1, i) = -1.0;
        b(2 * i) = 1.0;
        b(2 * i + 1) = 1.0;
    }
    A.row(2 * d).setOnes();     //  sum(x) <= eps
    A.row(2 * d + 1).setConstant(-1.0); // -sum(x) <= eps
    b(2 * d) = eps;
    b(2 * d + 1) = eps;
    x0 = Eigen::VectorXd::Zero(d);
}

// Build random dense polytope guaranteed to have interior
static void make_random_dense(int d, int m, Eigen::MatrixXd& A, Eigen::VectorXd& b, Eigen::VectorXd& x0, int seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> randn(0.0, 1.0);
    A.resize(m, d);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < d; ++j)
            A(i, j) = randn(rng);

    // x0 = 0 is interior; set b = A*0 + margin
    x0 = Eigen::VectorXd::Zero(d);
    b = Eigen::VectorXd::Constant(m, 1.0); // all rows have slack 1.0 at origin
}

static Eigen::MatrixXd run_chr(const Eigen::MatrixXd& A,
                                const Eigen::VectorXd& b,
                                const Eigen::VectorXd& x0,
                                int n_chains,
                                int n_samples,
                                int thinning,
                                int seed) {
    naja::gpu::DMatrix<double> A_d(A);
    naja::gpu::DVector<double> b_d(b);
    const int d = A.cols();
    Eigen::MatrixXd X0(d, n_chains);
    for (int c = 0; c < n_chains; ++c) X0.col(c) = x0;
    naja::gpu::DMatrix<double> X_d(X0);

    auto samples_d = naja::gpu::CoordinateHitAndRun(
        A_d, b_d, X_d,
        n_samples, thinning, n_chains,
        128, seed, 0.0, 0, 0.0, 8, 1,
        nullptr, nullptr, nullptr, nullptr
    );
    cudaDeviceSynchronize();
    return samples_d.toHost();
}

int main() {
    naja::gpu::set_device(0);

    const int n_chains = 4;
    const int n_samples = 5000;
    const int thinning = 10;
    const double feas_tol = 1e-7;
    int total_failures = 0;

    // Test 1: 10D hypercube
    {
        Eigen::MatrixXd A; Eigen::VectorXd b, x0;
        make_cube(10, A, b, x0);
        auto S = run_chr(A, b, x0, n_chains, n_samples, thinning, 42);
        total_failures += check_feasibility(A, b, S, feas_tol, "10D hypercube");
    }

    // Test 2: 8D simplex
    {
        Eigen::MatrixXd A; Eigen::VectorXd b, x0;
        make_simplex(8, A, b, x0);
        auto S = run_chr(A, b, x0, n_chains, n_samples, thinning, 123);
        total_failures += check_feasibility(A, b, S, feas_tol, "8D simplex");
    }

    // Test 3: 6D thin slab (eps=1e-4)
    {
        Eigen::MatrixXd A; Eigen::VectorXd b, x0;
        make_thin_slab(6, 1e-4, A, b, x0);
        auto S = run_chr(A, b, x0, n_chains, n_samples, thinning, 777);
        total_failures += check_feasibility(A, b, S, feas_tol, "6D thin slab eps=1e-4");
    }

    // Test 4: 15D random dense (60 constraints)
    {
        Eigen::MatrixXd A; Eigen::VectorXd b, x0;
        make_random_dense(15, 60, A, b, x0, 999);
        auto S = run_chr(A, b, x0, n_chains, n_samples, thinning, 555);
        total_failures += check_feasibility(A, b, S, feas_tol, "15D random dense 60 rows");
    }

    // Test 5: 30D random dense (120 constraints) - bigger stress test
    {
        Eigen::MatrixXd A; Eigen::VectorXd b, x0;
        make_random_dense(30, 120, A, b, x0, 2024);
        auto S = run_chr(A, b, x0, n_chains, n_samples, thinning, 333);
        total_failures += check_feasibility(A, b, S, feas_tol, "30D random dense 120 rows");
    }

    if (total_failures > 0) {
        std::cerr << "\nFAILED: " << total_failures << " total constraint violations\n";
        return 1;
    }
    std::cout << "\nAll feasibility tests passed.\n";
    return 0;
}


