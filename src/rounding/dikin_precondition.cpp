#include "rounding/dikin_precondition.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace naja::rounding {
namespace {

// Compute chord length through x_c along direction v in polytope A*x <= b.
static double chord_along(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                          const Eigen::VectorXd& x_c, const Eigen::VectorXd& v) {
    const Eigen::VectorXd slack = b - A * x_c;
    const Eigen::VectorXd Av = A * v;

    double t_min = -std::numeric_limits<double>::infinity();
    double t_max = std::numeric_limits<double>::infinity();

    for (int i = 0; i < A.rows(); ++i) {
        if (std::abs(Av[i]) < 1e-15) continue;
        double t = slack[i] / Av[i];
        if (Av[i] > 0.0) {
            t_max = std::min(t_max, t);
        } else {
            t_min = std::max(t_min, t);
        }
    }
    if (t_max <= t_min) return 0.0;
    return t_max - t_min;
}

} // namespace

DikinResult dikin_precondition(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& x_c,
    int k_directions,
    double tight_factor,
    double target_chord,
    double alpha_max) {

    const int m = A.rows();
    const int d = A.cols();
    if (m == 0 || d == 0) throw std::invalid_argument("dikin_precondition: empty polytope");

    // Step 1: Compute row norms and normalized distances to facets
    Eigen::VectorXd row_norms(m);
    Eigen::VectorXd deltas(m);
    Eigen::MatrixXd N(m, d);  // unit normals

    const Eigen::VectorXd slack = b - A * x_c;

    for (int i = 0; i < m; ++i) {
        row_norms[i] = A.row(i).norm();
        if (row_norms[i] < 1e-15) {
            deltas[i] = std::numeric_limits<double>::infinity();
            N.row(i).setZero();
        } else {
            deltas[i] = slack[i] / row_norms[i];
            N.row(i) = A.row(i) / row_norms[i];
        }
    }

    // Step 2: Select tight set T
    // Include all constraints with delta_i <= tight_factor * min_delta
    double min_delta = std::numeric_limits<double>::infinity();
    for (int i = 0; i < m; ++i) {
        if (deltas[i] > 0 && std::isfinite(deltas[i])) {
            min_delta = std::min(min_delta, deltas[i]);
        }
    }

    double threshold = tight_factor * min_delta;
    std::vector<int> tight_set;
    for (int i = 0; i < m; ++i) {
        if (deltas[i] > 0 && deltas[i] <= threshold) {
            tight_set.push_back(i);
        }
    }

    if (tight_set.empty()) {
        // No tight constraints — return identity transform
        DikinResult result;
        result.P = Eigen::MatrixXd::Identity(d, d);
        result.A_new = A;
        result.b_new = slack;  // shifted so x_c maps to origin
        result.x0_new = Eigen::VectorXd::Zero(d);
        result.n_corrected = 0;
        return result;
    }

    // Step 3: Build barrier Hessian H = sum_{i in T} (1/delta_i^2) n_i n_i'
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(d, d);
    for (int i : tight_set) {
        double w = 1.0 / (deltas[i] * deltas[i]);
        H.noalias() += w * N.row(i).transpose() * N.row(i);
    }

    // Step 4: Eigen-decompose H, take top-k eigenvectors
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(H);
    const Eigen::VectorXd& eigenvalues = eig.eigenvalues();
    const Eigen::MatrixXd& eigenvectors = eig.eigenvectors();

    // Eigenvalues are sorted ascending — we want the largest (last)
    int k = std::min(k_directions, d);
    // Only correct directions with eigenvalue significantly above the median
    double median_eval = eigenvalues[d / 2];
    int actual_k = 0;
    for (int j = d - 1; j >= 0 && actual_k < k; --j) {
        if (eigenvalues[j] > 10.0 * std::max(median_eval, 1e-10)) {
            ++actual_k;
        } else {
            break;
        }
    }
    if (actual_k == 0) actual_k = 1;

    Eigen::MatrixXd V_k(d, actual_k);
    for (int j = 0; j < actual_k; ++j) {
        V_k.col(j) = eigenvectors.col(d - 1 - j);
    }

    // Step 5: Compute actual chord lengths along pinched directions
    // Also compute median chord along coordinate axes for target
    if (target_chord <= 0.0) {
        std::vector<double> coord_chords;
        for (int j = 0; j < std::min(d, 50); ++j) {
            Eigen::VectorXd ej = Eigen::VectorXd::Zero(d);
            ej[j] = 1.0;
            double c = chord_along(A, b, x_c, ej);
            if (c > 0) coord_chords.push_back(c);
        }
        if (!coord_chords.empty()) {
            std::sort(coord_chords.begin(), coord_chords.end());
            target_chord = coord_chords[coord_chords.size() / 2];
        } else {
            target_chord = 1.0;
        }
    }

    Eigen::VectorXd alphas(actual_k);
    for (int j = 0; j < actual_k; ++j) {
        double chord_j = chord_along(A, b, x_c, V_k.col(j));
        if (chord_j < 1e-15) {
            // Dead direction — clip to modest expansion
            alphas[j] = std::min(alpha_max, 10.0);
        } else {
            double raw_alpha = target_chord / chord_j;
            alphas[j] = std::min(std::max(raw_alpha, 1.0), alpha_max);
        }
    }

    // Step 6: Build corrective transform P = I + V_k (diag(alpha) - I) V_k'
    // This stretches the pinched directions by alpha_j while leaving others unchanged.
    Eigen::MatrixXd P = Eigen::MatrixXd::Identity(d, d);
    for (int j = 0; j < actual_k; ++j) {
        P.noalias() += (alphas[j] - 1.0) * V_k.col(j) * V_k.col(j).transpose();
    }

    // P^{-1} = I + V_k (diag(1/alpha) - I) V_k'
    Eigen::MatrixXd P_inv = Eigen::MatrixXd::Identity(d, d);
    for (int j = 0; j < actual_k; ++j) {
        P_inv.noalias() += (1.0 / alphas[j] - 1.0) * V_k.col(j) * V_k.col(j).transpose();
    }

    // Step 7: Transform constraint system
    // In new coords z: y = P_inv * z + x_c
    // Constraint: A * y <= b  =>  A * (P_inv * z + x_c) <= b  =>  (A * P_inv) * z <= b - A * x_c
    Eigen::MatrixXd A_new = A * P_inv;
    Eigen::VectorXd b_new = b - A * x_c;

    DikinResult result;
    result.P = P;
    result.A_new = A_new;
    result.b_new = b_new;
    result.x0_new = Eigen::VectorXd::Zero(d);  // x_c maps to origin
    result.n_corrected = actual_k;

    return result;
}

} // namespace naja::rounding

