#pragma once
#include "dmatrix.h"
#include "dvector.h"

namespace coloring {
namespace gpu {

// Computes one row of the overlap matrix.
// samples: (n_samples, dims) - samples are rows, features are columns.
// sigmas: (dims) - scaling factors.
// target_indices: (n_targets) - indices of columns to check.
// row_O: (n_targets) - output average scores.
// p: exponent for soft score (default 3.0).
void compute_soft_overlap_row(
    const naja::gpu::DMatrix<double>& samples,
    const naja::gpu::DVector<double>& sigmas,
    const naja::gpu::DVector<int>& target_indices,
    naja::gpu::DVector<double>& row_O,
    double p = 3.0
);

// Computes Universality score for each sample.
// u_scores: (n_samples) - output U(v) for each sample.
void compute_universality(
    const naja::gpu::DMatrix<double>& samples,
    const naja::gpu::DVector<double>& sigmas,
    const naja::gpu::DVector<int>& target_indices,
    naja::gpu::DVector<double>& u_scores,
    double p = 3.0
);

}
}
