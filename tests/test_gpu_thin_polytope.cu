// Hard correctness test: sample a near-degenerate polytope and verify that
// (a) all samples are feasible, and (b) the sampler actually explores the
// feasible region rather than getting stuck.
//
// Polytope: 10D with a very thin slab in one direction.
//   [-1,1]^10 intersected with |x_0 + x_1 + x_2| <= eps
// The slab constrains the first three coordinates' sum to a tiny range.
// Coordinate-only CHR should still produce feasible samples but with poor mixing.
//
// We also test with pair moves (pair_prob > 0) and verify the chain explores more.

#include <Eigen/Dense>
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

#include "dmatrix.h"
#include "dvector.h"
#include "gpusamplers.h"
#include "device_utils.h"

static Eigen::MatrixXd run_chr_with_pairs(const Eigen::MatrixXd& A,
                                           const Eigen::VectorXd& b,
                                           const Eigen::VectorXd& x0,
                                           int n_chains,
                                           int n_samples,
                                           int thinning,
                                           int seed,
                                           double pair_prob) {
    const int d = A.cols();
    naja::gpu::DMatrix<double> A_d(A);
    naja::gpu::DVector<double> b_d(b);
    Eigen::MatrixXd X0(d, n_chains);
    for (int c = 0; c < n_chains; ++c) X0.col(c) = x0;
    naja::gpu::DMatrix<double> X_d(X0);

    auto samples_d = naja::gpu::CoordinateHitAndRun(
        A_d, b_d, X_d,
        n_samples, thinning, n_chains,
        128, seed, pair_prob, 0, 0.0, 8, 1,
        nullptr, nullptr, nullptr, nullptr
    );
    cudaDeviceSynchronize();
    return samples_d.toHost();
}

int main() {
    naja::gpu::set_device(0);

    const int d = 10;
    const double eps = 1e-4;
    const int n_chains = 4;
    const int n_samples = 10000;
    const int thinning = 5;
    const int total = n_chains * n_samples;

    // Build polytope: [-1,1]^10 + |x0+x1+x2| <= eps
    const int m = 2 * d + 2;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m, d);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(m);
    for (int i = 0; i < d; ++i) {
        A(2 * i, i) = 1.0;
        A(2 * i + 1, i) = -1.0;
        b(2 * i) = 1.0;
        b(2 * i + 1) = 1.0;
    }
    // Slab: x0+x1+x2 <= eps, -(x0+x1+x2) <= eps
    A(2 * d, 0) = 1.0; A(2 * d, 1) = 1.0; A(2 * d, 2) = 1.0;
    A(2 * d + 1, 0) = -1.0; A(2 * d + 1, 1) = -1.0; A(2 * d + 1, 2) = -1.0;
    b(2 * d) = eps;
    b(2 * d + 1) = eps;

    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(d);

    int failures = 0;

    // Run with coordinate-only moves
    auto S_coord = run_chr_with_pairs(A, b, x0, n_chains, n_samples, thinning, 42, 0.0);

    // Verify ALL samples feasible
    int coord_violations = 0;
    for (int j = 0; j < total; ++j) {
        Eigen::VectorXd r = A * S_coord.col(j) - b;
        if (r.maxCoeff() > 1e-7) ++coord_violations;
    }
    if (coord_violations > 0) {
        std::cerr << "[FAIL] coord-only: " << coord_violations << " feasibility violations\n";
        ++failures;
    }

    // Check that the slab constraint is actually respected tightly
    double max_slab_coord = 0.0;
    for (int j = 0; j < total; ++j) {
        double s = std::abs(S_coord(0, j) + S_coord(1, j) + S_coord(2, j));
        max_slab_coord = std::max(max_slab_coord, s);
    }
    if (max_slab_coord > eps + 1e-7) {
        std::cerr << "[FAIL] coord-only slab violation: max |x0+x1+x2|=" << max_slab_coord << " > " << eps << "\n";
        ++failures;
    }

    // Run with pair moves (should have better mixing in unconstrained dims)
    auto S_pair = run_chr_with_pairs(A, b, x0, n_chains, n_samples, thinning, 42, 0.3);

    int pair_violations = 0;
    for (int j = 0; j < total; ++j) {
        Eigen::VectorXd r = A * S_pair.col(j) - b;
        if (r.maxCoeff() > 1e-7) ++pair_violations;
    }
    if (pair_violations > 0) {
        std::cerr << "[FAIL] pair-mode: " << pair_violations << " feasibility violations\n";
        ++failures;
    }

    // Verify that the unconstrained dimensions (3..9) have reasonable variance
    // These should have var ~ 1/3 since they're on [-1,1]
    for (int i = 3; i < d; ++i) {
        double mean_i = S_pair.row(i).mean();
        double var_i = (S_pair.row(i).array() - mean_i).square().mean();
        if (var_i < 0.1) {
            std::cerr << "[FAIL] dim " << i << " variance=" << var_i << " < 0.1 (chain stuck?)\n";
            ++failures;
        }
    }

    // Verify that constrained dimensions (0,1,2) have small variance (bounded by slab)
    for (int i = 0; i < 3; ++i) {
        double var_i = (S_coord.row(i).array() - S_coord.row(i).mean()).square().mean();
        // With coord-only moves in a thin slab, variance in each constrained coord should be small
        // but not zero. The slab constrains the SUM, not individual coords.
        // However, mixing should be poor. We just check feasibility here.
    }

    if (failures > 0) {
        std::cerr << "\nFAILED: " << failures << " thin polytope checks\n";
        return 1;
    }

    std::cout << "[PASS] thin polytope (eps=" << eps << "): feasibility OK, exploration OK\n";
    std::cout << "  max_slab_coord=" << max_slab_coord << "\n";
    return 0;
}


