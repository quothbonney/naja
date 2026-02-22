#include "rounding/barrier_rotation.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

namespace naja::rounding {

BarrierRotation compute_barrier_rotation(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& x_c,
    bool verbose) {

    const int m = static_cast<int>(A.rows());
    const int d = static_cast<int>(A.cols());

    // 1. Slack, clamped so one near-tight facet can't dominate the entire Hessian.
    Eigen::VectorXd slack = b - A * x_c;

    // tau = 1e-6 * median(positive slacks)
    std::vector<double> pos_slacks;
    pos_slacks.reserve(m);
    for (int i = 0; i < m; ++i) {
        if (slack[i] > 0.0) pos_slacks.push_back(slack[i]);
    }
    double tau;
    if (!pos_slacks.empty()) {
        auto mid = pos_slacks.begin() + static_cast<long>(pos_slacks.size()) / 2;
        std::nth_element(pos_slacks.begin(), mid, pos_slacks.end());
        tau = 1e-6 * (*mid);
    } else {
        tau = 1e-12;
    }

    int n_clamped = 0;
    for (int i = 0; i < m; ++i) {
        if (slack[i] < tau) {
            slack[i] = tau;
            ++n_clamped;
        }
    }

    // 2. H = A' diag(1/s^2) A  via  B = diag(1/s) A,  H = B'B.
    Eigen::VectorXd inv_slack = slack.cwiseInverse();
    Eigen::MatrixXd B = inv_slack.asDiagonal() * A;   // (m, d)
    Eigen::MatrixXd H(d, d);
    H.noalias() = B.transpose() * B;                  // (d, d)

    // 3. Tiny ridge for numerical stability.
    double trace_H = H.trace();
    double ridge = 1e-10 * trace_H / d;
    H.diagonal().array() += ridge;

    // 4. Eigen-decompose (self-adjoint → guaranteed real, orthogonal Q).
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(H);

    BarrierRotation result;
    result.Q = eig.eigenvectors();          // (d, d), columns = eigenvectors
    result.eigenvalues = eig.eigenvalues(); // (d,),   ascending

    if (verbose) {
        // Diagonalization sanity check: ||off-diag(Q'HQ)|| / ||Q'HQ||
        Eigen::MatrixXd D = result.Q.transpose() * H * result.Q;
        double off_diag_sq = 0.0;
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j)
                if (i != j) off_diag_sq += D(i, j) * D(i, j);
        double total_sq = D.squaredNorm();
        double diag_quality = (total_sq > 0.0) ? std::sqrt(off_diag_sq / total_sq) : 0.0;

        double eig_min = result.eigenvalues[0];
        double eig_max = result.eigenvalues[d - 1];
        double eig_med = result.eigenvalues[d / 2];

        std::cout << "  barrier_rotation  dim=" << d
                  << " clamped=" << n_clamped
                  << " ridge=" << std::scientific << std::setprecision(2) << ridge
                  << std::fixed
                  << "\n    eigenvalues: min=" << std::setprecision(4) << eig_min
                  << " med=" << eig_med
                  << " max=" << eig_max
                  << " condition=" << std::setprecision(1) << (eig_max / std::max(eig_min, 1e-30))
                  << "\n    ||off-diag|| / ||D|| = " << std::scientific << std::setprecision(3) << diag_quality
                  << std::fixed << "\n";
    }

    return result;
}

} // namespace naja::rounding

