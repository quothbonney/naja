// Hard correctness test: sample the hypercube [-1,1]^d and verify that
// sample moments match the known analytical values for a uniform distribution.
//
// Analytical:
//   E[x_i] = 0
//   Var[x_i] = 1/3
//   Cov[x_i, x_j] = 0  for i != j
//
// We run enough samples that deviations indicate a real bug, not noise.

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

    const int d = 8;
    const int n_chains = 8;
    const int n_samples = 20000;
    const int thinning = 20; // generous thinning for decorrelation
    const int total = n_chains * n_samples;

    // Build [-1,1]^d
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2 * d, d);
    Eigen::VectorXd b = Eigen::VectorXd::Ones(2 * d);
    for (int i = 0; i < d; ++i) {
        A(2 * i, i) = 1.0;
        A(2 * i + 1, i) = -1.0;
    }
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(d);

    naja::gpu::DMatrix<double> A_d(A);
    naja::gpu::DVector<double> b_d(b);
    Eigen::MatrixXd X0(d, n_chains);
    for (int c = 0; c < n_chains; ++c) X0.col(c) = x0;
    naja::gpu::DMatrix<double> X_d(X0);

    auto samples_d = naja::gpu::CoordinateHitAndRun(
        A_d, b_d, X_d,
        n_samples, thinning, n_chains,
        128, 42, 0.0, 0, 0.0, 8, 1,
        nullptr, nullptr, nullptr, nullptr
    );
    cudaDeviceSynchronize();
    Eigen::MatrixXd S = samples_d.toHost(); // d x total

    if (S.cols() != total) {
        std::cerr << "unexpected sample count: " << S.cols() << " != " << total << "\n";
        return 1;
    }

    // Compute per-coordinate statistics
    const double expected_mean = 0.0;
    const double expected_var = 1.0 / 3.0;

    // Tolerance: with total samples and some autocorrelation, allow generous margins.
    // Standard error of mean: sqrt(var/N_eff). Even with ESS = total/10, SE ~ 0.006.
    // We use 0.05 as a very generous threshold that should never false-positive.
    const double mean_tol = 0.05;
    // Variance has higher uncertainty; use relative tolerance of 15%.
    const double var_rel_tol = 0.15;

    int failures = 0;

    for (int i = 0; i < d; ++i) {
        const double mean_i = S.row(i).mean();
        const double var_i = (S.row(i).array() - mean_i).square().mean();

        if (std::abs(mean_i - expected_mean) > mean_tol) {
            std::cerr << "[FAIL] dim " << i << ": mean=" << mean_i
                      << ", expected ~" << expected_mean
                      << ", error=" << std::abs(mean_i - expected_mean) << "\n";
            ++failures;
        }

        if (std::abs(var_i - expected_var) / expected_var > var_rel_tol) {
            std::cerr << "[FAIL] dim " << i << ": var=" << var_i
                      << ", expected ~" << expected_var
                      << ", rel_error=" << std::abs(var_i - expected_var) / expected_var << "\n";
            ++failures;
        }
    }

    // Cross-coordinate independence: max absolute off-diagonal correlation should be small.
    Eigen::MatrixXd centered = S;
    for (int i = 0; i < d; ++i) {
        centered.row(i).array() -= S.row(i).mean();
    }
    Eigen::MatrixXd cov = (centered * centered.transpose()) / (double)(total - 1);
    double max_abs_corr = 0.0;
    for (int i = 0; i < d; ++i) {
        for (int j = i + 1; j < d; ++j) {
            double corr = cov(i, j) / std::sqrt(cov(i, i) * cov(j, j));
            max_abs_corr = std::max(max_abs_corr, std::abs(corr));
        }
    }
    // With large N and thinning, off-diagonal correlations should be small.
    const double corr_tol = 0.05;
    if (max_abs_corr > corr_tol) {
        std::cerr << "[FAIL] max off-diagonal |correlation|=" << max_abs_corr
                  << " > " << corr_tol << "\n";
        ++failures;
    }

    if (failures > 0) {
        std::cerr << "\nFAILED: " << failures << " moment checks failed for " << d << "D cube\n";
        return 1;
    }

    std::cout << "[PASS] " << d << "D cube moments: mean, variance, correlation all within tolerance\n";
    std::cout << "  total_samples=" << total << " thinning=" << thinning << "\n";
    std::cout << "  max_abs_corr=" << max_abs_corr << "\n";
    return 0;
}


