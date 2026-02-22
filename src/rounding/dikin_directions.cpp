#include "rounding/dikin_directions.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace naja::rounding {

DikinDirections compute_dikin_directions(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& x_c,
    int n_base_rows,
    int max_directions,
    double tight_factor,
    double eigenvalue_ratio) {

    const int m = A.rows();
    const int d = A.cols();
    DikinDirections result;
    result.k = 0;

    if (n_base_rows >= m) return result;  // no extra rows

    const Eigen::VectorXd slack = b - A * x_c;

    // Compute normalized distances for extra rows only (delta barrier Hessian).
    // This targets ONLY the damage introduced by the KO/conditioning constraints,
    // not pre-existing base rounding imperfections.
    const int n_extra = m - n_base_rows;
    Eigen::VectorXd deltas(n_extra);
    Eigen::MatrixXd normals(n_extra, d);

    double min_delta = std::numeric_limits<double>::infinity();
    for (int i = 0; i < n_extra; ++i) {
        int row = n_base_rows + i;
        double norm = A.row(row).norm();
        if (norm < 1e-15) {
            deltas[i] = std::numeric_limits<double>::infinity();
            normals.row(i).setZero();
        } else {
            deltas[i] = slack[row] / norm;
            normals.row(i) = A.row(row) / norm;
        }
        if (deltas[i] > 0 && std::isfinite(deltas[i])) {
            min_delta = std::min(min_delta, deltas[i]);
        }
    }

    if (!std::isfinite(min_delta) || min_delta <= 0) return result;

    // Build delta-Hessian: H_delta = sum_{tight extra rows} (1/delta_i^2) n_i n_i'
    double threshold = tight_factor * min_delta;
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(d, d);
    int n_tight = 0;
    for (int i = 0; i < n_extra; ++i) {
        if (deltas[i] > 0 && deltas[i] <= threshold) {
            double w = 1.0 / (deltas[i] * deltas[i]);
            H.noalias() += w * normals.row(i).transpose() * normals.row(i);
            ++n_tight;
        }
    }

    if (n_tight == 0) return result;

    // Eigen-decompose and take top-k
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(H);
    const auto& eigenvalues = eig.eigenvalues();
    const auto& eigenvectors = eig.eigenvectors();

    // Select directions with eigenvalue significantly above noise.
    // eigenvalue_ratio=10 → strict (delta mode), =1 → always take max_directions.
    double median_eval = eigenvalues[d / 2];
    int k = 0;
    for (int j = d - 1; j >= 0 && k < max_directions; --j) {
        if (eigenvalues[j] > eigenvalue_ratio * std::max(median_eval, 1e-10)) {
            ++k;
        } else {
            break;
        }
    }
    if (k == 0) k = std::min(1, max_directions);

    // Extract directions (columns of V) and precompute A*V
    result.V.resize(d, k);
    for (int j = 0; j < k; ++j) {
        result.V.col(j) = eigenvectors.col(d - 1 - j);
    }
    result.Av = A * result.V;  // (m, k)
    result.k = k;

    return result;
}

} // namespace naja::rounding

