#pragma once

#include <Eigen/Dense>

namespace naja::validate {

struct EssResult {
    Eigen::VectorXf ess;           // per-dim ESS (min across chains)
    float min, p10, median, max;
};

struct RhatResult {
    Eigen::VectorXf rhat;          // per-dim split-R-hat
    float median, max;
    int above_1_1, above_1_2;
};

// Compute per-dimension effective sample size via FFT-based autocovariance
// (Geyer's initial positive sequence estimator). Returns the min ESS across
// chains for each dimension.
//
// samples: (dim, n_total) column-major, naja CHR output order:
// columns are interleaved by saved sample, i.e. col = sample_idx*n_chains + chain.
EssResult compute_ess(const Eigen::MatrixXf& samples, int n_chains);

// Compute split-R-hat convergence diagnostic. Each chain is split in half,
// giving 2*n_chains half-chains. R-hat is sqrt(var_hat / W) per dimension.
//
// samples: (dim, n_total) column-major, naja CHR output order:
// columns are interleaved by saved sample, i.e. col = sample_idx*n_chains + chain.
RhatResult compute_split_rhat(const Eigen::MatrixXf& samples, int n_chains);

} // namespace naja::validate

