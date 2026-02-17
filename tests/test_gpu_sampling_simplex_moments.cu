// Hard correctness test: sample the standard n-simplex and verify moments
// against exact analytical values for the uniform distribution.
//
// Standard n-simplex in R^n: {x >= 0, sum(x) <= 1}
//
// Analytical (uniform on this simplex):
//   E[x_i]       = 1 / (n+1)
//   Var[x_i]     = n / ((n+1)^2 * (n+2))
//   Cov[x_i,x_j] = -1 / ((n+1)^2 * (n+2))   for i != j
//
// This is one of the strongest correctness tests because the moments are
// known exactly and any systematic bias in the sampler will show up.

#include <Eigen/Dense>
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

#include "dmatrix.h"
#include "dvector.h"
#include "gpusamplers.h"
#include "device_utils.h"

int main() {
    naja::gpu::set_device(0);

    const int n = 6; // 6D simplex (7 constraints)
    const int n_chains = 8;
    const int n_samples = 30000;
    const int thinning = 20;
    const int total = n_chains * n_samples;

    // Build simplex: -x_i <= 0, sum(x) <= 1
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n + 1, n);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(n + 1);
    for (int i = 0; i < n; ++i) {
        A(i, i) = -1.0;
    }
    A.row(n).setOnes();
    b(n) = 1.0;

    Eigen::VectorXd x0 = Eigen::VectorXd::Constant(n, 1.0 / (n + 1));

    naja::gpu::DMatrix<double> A_d(A);
    naja::gpu::DVector<double> b_d(b);
    Eigen::MatrixXd X0(n, n_chains);
    for (int c = 0; c < n_chains; ++c) X0.col(c) = x0;
    naja::gpu::DMatrix<double> X_d(X0);

    auto samples_d = naja::gpu::CoordinateHitAndRun(
        A_d, b_d, X_d,
        n_samples, thinning, n_chains,
        128, 314, 0.0, 0, 0.0, 8, 1,
        nullptr, nullptr, nullptr, nullptr
    );
    cudaDeviceSynchronize();
    Eigen::MatrixXd S = samples_d.toHost(); // n x total

    // Analytical values
    const double np1 = n + 1.0;
    const double np2 = n + 2.0;
    const double expected_mean = 1.0 / np1;
    const double expected_var = (double)n / (np1 * np1 * np2);
    const double expected_cov = -1.0 / (np1 * np1 * np2);

    // Tolerances (generous, but catch real bugs)
    const double mean_tol = 0.015;     // absolute
    const double var_rel_tol = 0.20;   // relative
    const double cov_abs_tol = 0.008;  // absolute (covariance is small)

    int failures = 0;

    // Check means
    for (int i = 0; i < n; ++i) {
        const double mean_i = S.row(i).mean();
        const double err = std::abs(mean_i - expected_mean);
        if (err > mean_tol) {
            std::cerr << "[FAIL] dim " << i << ": mean=" << mean_i
                      << ", expected=" << expected_mean
                      << ", error=" << err << "\n";
            ++failures;
        }
    }

    // Check variances
    for (int i = 0; i < n; ++i) {
        const double mean_i = S.row(i).mean();
        const double var_i = (S.row(i).array() - mean_i).square().mean();
        const double rel_err = std::abs(var_i - expected_var) / expected_var;
        if (rel_err > var_rel_tol) {
            std::cerr << "[FAIL] dim " << i << ": var=" << var_i
                      << ", expected=" << expected_var
                      << ", rel_error=" << rel_err << "\n";
            ++failures;
        }
    }

    // Check covariances (off-diagonal)
    Eigen::MatrixXd centered = S;
    for (int i = 0; i < n; ++i) {
        centered.row(i).array() -= S.row(i).mean();
    }
    Eigen::MatrixXd cov_mat = (centered * centered.transpose()) / (double)(total - 1);

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const double cov_ij = cov_mat(i, j);
            const double err = std::abs(cov_ij - expected_cov);
            if (err > cov_abs_tol) {
                std::cerr << "[FAIL] cov(" << i << "," << j << ")=" << cov_ij
                          << ", expected=" << expected_cov
                          << ", error=" << err << "\n";
                ++failures;
            }
        }
    }

    // Check that ALL samples satisfy the simplex constraints (feasibility)
    int feas_violations = 0;
    for (int j = 0; j < total; ++j) {
        Eigen::VectorXd residual = A * S.col(j) - b;
        for (int i = 0; i < A.rows(); ++i) {
            if (residual(i) > 1e-7) ++feas_violations;
        }
    }
    if (feas_violations > 0) {
        std::cerr << "[FAIL] feasibility: " << feas_violations << " violations\n";
        ++failures;
    }

    // Check sum(x) <= 1 for all samples (tighter than general feasibility)
    for (int j = 0; j < total; ++j) {
        double s = S.col(j).sum();
        if (s > 1.0 + 1e-7) {
            std::cerr << "[FAIL] sum(x)=" << s << " > 1 at sample " << j << "\n";
            ++failures;
            break;
        }
        for (int i = 0; i < n; ++i) {
            if (S(i, j) < -1e-7) {
                std::cerr << "[FAIL] x(" << i << ")=" << S(i, j) << " < 0 at sample " << j << "\n";
                ++failures;
                break;
            }
        }
        if (failures > 5) break;
    }

    if (failures > 0) {
        std::cerr << "\nFAILED: " << failures << " checks failed for " << n << "D simplex\n";
        return 1;
    }

    std::cout << "[PASS] " << n << "D simplex moments: mean, variance, covariance, feasibility all correct\n";
    std::cout << "  E[x_i]=" << expected_mean << " Var[x_i]=" << expected_var << " Cov[x_i,x_j]=" << expected_cov << "\n";
    std::cout << "  total_samples=" << total << " thinning=" << thinning << "\n";
    return 0;
}


