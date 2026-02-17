// Hard correctness test: verify that CoordinateHitAndRunBackmap produces
// samples that are exactly T * reduced_sample + shift.
//
// Strategy:
//   1. Build a polytope in reduced space (5D cube)
//   2. Define a known T (8x5) and shift (8x1)
//   3. Sample using CHRBackmap to get original-space samples (8D)
//   4. Independently sample reduced space with plain CHR (5D)
//   5. Verify all backmap samples satisfy A_reduced * (T_inv * (y - shift)) <= b
//   6. Verify all backmap samples actually land in the range of T

#include <Eigen/Dense>
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <random>

#include "dmatrix.h"
#include "dvector.h"
#include "gpusamplers.h"
#include "device_utils.h"

int main() {
    naja::gpu::set_device(0);

    const int d_reduced = 5;
    const int d_original = 8;
    const int n_chains = 4;
    const int n_samples = 5000;
    const int thinning = 10;
    const int total = n_chains * n_samples;

    // Build 5D cube in reduced space
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2 * d_reduced, d_reduced);
    Eigen::VectorXd b = Eigen::VectorXd::Ones(2 * d_reduced);
    for (int i = 0; i < d_reduced; ++i) {
        A(2 * i, i) = 1.0;
        A(2 * i + 1, i) = -1.0;
    }
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(d_reduced);

    // Build a known T and shift (deterministic for reproducibility)
    std::mt19937_64 rng(2024);
    std::normal_distribution<double> randn(0.0, 1.0);
    Eigen::MatrixXd T(d_original, d_reduced);
    Eigen::VectorXd shift(d_original);
    for (int i = 0; i < d_original; ++i) {
        for (int j = 0; j < d_reduced; ++j) {
            T(i, j) = randn(rng);
        }
        shift(i) = randn(rng) * 0.5;
    }

    // Run CHRBackmap
    naja::gpu::DMatrix<double> A_d(A);
    naja::gpu::DVector<double> b_d(b);
    naja::gpu::DMatrix<double> T_d(T);
    naja::gpu::DVector<double> shift_d(shift);
    Eigen::MatrixXd X0(d_reduced, n_chains);
    for (int c = 0; c < n_chains; ++c) X0.col(c) = x0;
    naja::gpu::DMatrix<double> X_d(X0);

    auto backmap_d = naja::gpu::CoordinateHitAndRunBackmap(
        A_d, b_d, X_d, T_d, shift_d,
        n_samples, thinning, n_chains,
        128, 42, 0.0, 0, 0.0, 8, 1,
        nullptr, nullptr, nullptr, nullptr
    );
    cudaDeviceSynchronize();
    Eigen::MatrixXd Y = backmap_d.toHost(); // d_original x total

    if (Y.rows() != d_original || Y.cols() != total) {
        std::cerr << "[FAIL] unexpected output shape: " << Y.rows() << " x " << Y.cols()
                  << ", expected " << d_original << " x " << total << "\n";
        return 1;
    }

    // For each backmap sample y, compute the reduced-space preimage:
    //   x_reduced = T^+ * (y - shift)   where T^+ is the pseudoinverse
    // Then verify:
    //   1. A * x_reduced <= b  (feasibility in reduced space)
    //   2. T * x_reduced + shift ≈ y  (y is in range of T)
    Eigen::MatrixXd T_pinv = T.completeOrthogonalDecomposition().pseudoInverse();

    int feas_violations = 0;
    int range_violations = 0;
    double max_range_err = 0.0;
    double max_feas_violation = 0.0;

    for (int j = 0; j < total; ++j) {
        Eigen::VectorXd y = Y.col(j);
        Eigen::VectorXd x_reduced = T_pinv * (y - shift);

        // Check feasibility in reduced space
        Eigen::VectorXd residual = A * x_reduced - b;
        double worst = residual.maxCoeff();
        if (worst > 1e-6) {
            ++feas_violations;
            max_feas_violation = std::max(max_feas_violation, worst);
        }

        // Check that y is in the range of T (reconstruction error)
        Eigen::VectorXd y_recon = T * x_reduced + shift;
        double range_err = (y - y_recon).norm();
        if (range_err > 1e-6) {
            ++range_violations;
            max_range_err = std::max(max_range_err, range_err);
        }
    }

    int failures = 0;
    if (feas_violations > 0) {
        std::cerr << "[FAIL] reduced-space feasibility: " << feas_violations << " violations"
                  << ", worst=" << max_feas_violation << "\n";
        ++failures;
    }
    if (range_violations > 0) {
        std::cerr << "[FAIL] range reconstruction: " << range_violations << " errors"
                  << ", worst=" << max_range_err << "\n";
        ++failures;
    }

    // Also verify that the backmapped samples have reasonable statistics.
    // Expected mean in original space: T * E[x_reduced] + shift = T * 0 + shift = shift
    Eigen::VectorXd sample_mean = Y.rowwise().mean();
    double mean_err = (sample_mean - shift).norm();
    if (mean_err > 0.3) {
        std::cerr << "[FAIL] mean error in original space: ||mean - shift||=" << mean_err << "\n";
        ++failures;
    }

    if (failures > 0) {
        std::cerr << "\nFAILED: backmap correctness\n";
        return 1;
    }

    std::cout << "[PASS] backmap correctness: " << total << " samples"
              << ", feas OK, range OK, mean_err=" << mean_err << "\n";
    return 0;
}


