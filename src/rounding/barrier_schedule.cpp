#include "rounding/barrier_schedule.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "rounding/jacobi.h"

namespace naja::rounding {

PairSchedule compute_barrier_pair_schedule(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& x_c,
    bool verbose) {

    const int m = static_cast<int>(A.rows());
    const int d = static_cast<int>(A.cols());

    // Slack and row-normalized weights.
    // delta_r = slack_r / ||a_r|| = Euclidean distance to facet r.
    // Weight = 1/delta_r^2 = ||a_r||^2 / slack_r^2.
    // We form B such that B_r = (1/delta_r) * a_r / ||a_r||  (unit normal, weighted by 1/delta).
    // Then G = B'B = sum_r (1/delta_r^2) n_r n_r' — the row-normalized barrier Hessian.

    Eigen::VectorXd slack = b - A * x_c;

    // Build sqrt(w) vector: sqrt(w_r) = ||a_r|| / slack_r = 1/delta_r
    // But we normalize rows first, so B_r = sqrt(w_r) * (a_r / ||a_r||) = (1/delta_r) * n_r.
    // Equivalently: B_r = a_r / slack_r  (since 1/delta_r * a_r/||a_r|| = a_r / slack_r).
    // So B = diag(1/slack) * A, and G = A' diag(1/slack^2) A.
    // BUT: this is not row-normalized (large ||a_r|| rows dominate).
    //
    // Correct row-normalized version:
    //   B_r = n_r / delta_r = (a_r / ||a_r||) * (||a_r|| / slack_r) = a_r / slack_r
    // Wait — that's the same thing. The key insight: n_r = a_r/||a_r||, so
    //   (1/delta_r^2) n_r n_r' = (||a_r||^2/slack_r^2) (a_r a_r') / ||a_r||^2 = a_r a_r' / slack_r^2
    // The row normalization cancels. So G = A' diag(1/s^2) A is already the
    // row-normalized barrier metric, because the unit normals absorb the ||a_r||.
    //
    // This is convenient: just scale each row of A by 1/s_r.

    Eigen::VectorXd inv_slack(m);
    for (int r = 0; r < m; ++r) {
        if (slack[r] > 0.0) {
            inv_slack[r] = 1.0 / slack[r];
        } else {
            inv_slack[r] = 0.0;  // tight or violated: skip
        }
    }

    // G = A' diag(1/s^2) A = B'B where B = diag(1/s) * A
    // Eigen: G = (inv_slack.asDiagonal() * A).transpose() * (inv_slack.asDiagonal() * A)
    // For efficiency, use the .noalias() path.
    Eigen::MatrixXd B = inv_slack.asDiagonal() * A;  // (m, d)
    Eigen::MatrixXd G(d, d);
    G.noalias() = B.transpose() * B;                 // (d, d)

    // Diagonal for correlation computation
    Eigen::VectorXd diag = G.diagonal();

    // Greedy matching by barrier-metric correlation |G_ij| / sqrt(G_ii G_jj)
    struct Candidate {
        double rho;
        int i, j;
    };

    // Only keep candidates with nonzero diagonal entries
    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<size_t>(d) * (d - 1) / 2);

    for (int i = 0; i < d; ++i) {
        if (diag[i] <= 0.0) continue;
        for (int j = i + 1; j < d; ++j) {
            if (diag[j] <= 0.0) continue;
            double rho = std::abs(G(i, j)) / std::sqrt(diag[i] * diag[j]);
            if (rho > 1e-12) {
                candidates.push_back({rho, i, j});
            }
        }
    }

    // Sort descending by correlation magnitude
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.rho > b.rho; });

    const int n_pairs = d / 2;
    PairSchedule sched;
    sched.i.resize(n_pairs);
    sched.j.resize(n_pairs);
    sched.c.resize(n_pairs);
    sched.s.resize(n_pairs);

    std::vector<bool> used(d, false);
    int matched = 0;

    for (const auto& c : candidates) {
        if (matched >= n_pairs) break;
        if (used[c.i] || used[c.j]) continue;

        used[c.i] = true;
        used[c.j] = true;

        // Jacobi angle from the 2x2 block of G.
        // jacobi_rotation_cs expects (var_i, var_j, cov_ij) and returns (cos, sin).
        // G_ii and G_jj play the role of variances, G_ij the covariance.
        auto cs = jacobi_rotation_cs(G(c.i, c.i), G(c.j, c.j), G(c.i, c.j));

        sched.i[matched] = c.i;
        sched.j[matched] = c.j;
        sched.c[matched] = cs.first;
        sched.s[matched] = cs.second;
        ++matched;
    }

    // Pair any remaining unmatched coordinates (weakly coupled — identity rotation)
    std::vector<int> unpaired;
    unpaired.reserve(d);
    for (int i = 0; i < d; ++i) {
        if (!used[i]) unpaired.push_back(i);
    }
    for (size_t k = 0; k + 1 < unpaired.size() && matched < n_pairs; k += 2) {
        int ii = unpaired[k];
        int jj = unpaired[k + 1];
        auto cs = jacobi_rotation_cs(G(ii, ii), G(jj, jj), G(ii, jj));
        sched.i[matched] = ii;
        sched.j[matched] = jj;
        sched.c[matched] = cs.first;
        sched.s[matched] = cs.second;
        ++matched;
    }

    // Trim if fewer than n_pairs (odd dimension leaves one unpaired)
    if (matched < n_pairs) {
        sched.i.conservativeResize(matched);
        sched.j.conservativeResize(matched);
        sched.c.conservativeResize(matched);
        sched.s.conservativeResize(matched);
    }

    if (verbose) {
        double max_rho = candidates.empty() ? 0.0 : candidates[0].rho;
        double p10_rho = candidates.empty() ? 0.0 : candidates[std::min((size_t)(candidates.size() / 10), candidates.size() - 1)].rho;
        std::cout << "  barrier_schedule " << matched << " pairs"
                  << " (max_rho=" << std::setprecision(4) << max_rho
                  << ", p10_rho=" << p10_rho << ")\n";
    }

    return sched;
}

} // namespace naja::rounding

