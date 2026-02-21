#pragma once

#include <Eigen/Dense>

namespace naja::validate {

struct BoundsResult {
    int n_violations;
    int n_checked;       // n_samples * n_reactions
    double worst_violation;
};

// Check samples (reduced space) against gem bounds via backmap v = T*y + shift.
// Subsamples for speed — checks up to max_samples columns.
BoundsResult check_bounds_backmap(
    const Eigen::MatrixXf& samples,   // (reduced_dim, n_total_samples) col-major
    const Eigen::MatrixXd& T,         // (n_reactions, reduced_dim)
    const Eigen::VectorXd& shift,     // (n_reactions)
    const Eigen::VectorXd& lb,        // (n_reactions)
    const Eigen::VectorXd& ub,        // (n_reactions)
    int max_samples = 1000);

} // namespace naja::validate

