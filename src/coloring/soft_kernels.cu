#include "soft_kernels.cuh"
#include "helper.h"
#include <cmath>

namespace coloring {
namespace gpu {

// Block reduction for summing scores
template <typename T>
__device__ T blockReduceSum(T val) {
    static __shared__ T shared[32]; // For warp reductions
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
    if (wid == 0) {
        for (int offset = 16; offset > 0; offset /= 2)
            val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel: One block per target
__global__ void soft_overlap_kernel(
    const double* __restrict__ samples, // (n_samples, dims)
    const double* __restrict__ target_sigmas,  // (n_targets)
    const int* __restrict__ target_indices, // (n_targets)
    double* __restrict__ row_O, // (n_targets)
    int n_samples,
    int dims,
    double p
) {
    int target_idx = blockIdx.x; // One block per target
    int tid = threadIdx.x;
    int stride = blockDim.x;

    int col_idx = target_indices[target_idx];
    double sigma = target_sigmas[target_idx] + 1e-12; // Avoid div/0

    double sum = 0.0;

    int offset = col_idx * n_samples;
    
    for (int i = tid; i < n_samples; i += stride) {
        double val = std::abs(samples[i + offset]);
        double normed = val / sigma;
        double score = exp(-pow(normed, p));
        sum += score;
    }

    sum = blockReduceSum(sum);

    if (tid == 0) {
        row_O[target_idx] = sum / n_samples;
    }
}

// Kernel: Compute U(v) for each sample
// Each thread handles one sample (or loop)
// Iterates over all targets
__global__ void universality_kernel(
    const double* __restrict__ samples, // (n_samples, dims)
    const double* __restrict__ target_sigmas,  // (n_targets)
    const int* __restrict__ target_indices, // (n_targets)
    double* __restrict__ u_scores, // (n_samples)
    int n_samples,
    int dims,
    int n_targets,
    double p
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_samples) return;

    double sum_score = 0.0;

    for (int j = 0; j < n_targets; ++j) {
        int col_idx = target_indices[j];
        double sigma = target_sigmas[j] + 1e-12;
        
        // Access sample 'idx' for column 'col_idx'
        // samples is (n_samples, dims) col-major? 
        // No, in main.cu we transposed it to be (n_samples, dims).
        // BUT we constructed DMatrix from it.
        // DMatrix is col-major.
        // So (n_samples, dims) matrix stores data as:
        // samples[idx + col * n_samples]
        
        int offset = col_idx * n_samples;
        double val = std::abs(samples[idx + offset]);
        double normed = val / sigma;
        sum_score += exp(-pow(normed, p));
    }

    u_scores[idx] = sum_score / n_targets;
}

void compute_soft_overlap_row(
    const naja::gpu::DMatrix<double>& samples,
    const naja::gpu::DVector<double>& sigmas,
    const naja::gpu::DVector<int>& target_indices,
    naja::gpu::DVector<double>& row_O,
    double p
) {
    int n_samples = samples.rows;
    int dims = samples.cols;
    int n_targets = target_indices.len;

    dim3 grid(n_targets);
    dim3 block(256);

    soft_overlap_kernel<<<grid, block>>>(
        samples.dmat,
        sigmas.dvec,
        target_indices.dvec,
        row_O.dvec,
        n_samples,
        dims,
        p
    );
    
    CUDA_CHECK(cudaGetLastError());
}

void compute_universality(
    const naja::gpu::DMatrix<double>& samples,
    const naja::gpu::DVector<double>& sigmas,
    const naja::gpu::DVector<int>& target_indices,
    naja::gpu::DVector<double>& u_scores,
    double p
) {
    int n_samples = samples.rows;
    int dims = samples.cols;
    int n_targets = target_indices.len;
    
    int threads = 256;
    int blocks = (n_samples + threads - 1) / threads;
    
    universality_kernel<<<blocks, threads>>>(
        samples.dmat,
        sigmas.dvec,
        target_indices.dvec,
        u_scores.dvec,
        n_samples,
        dims,
        n_targets,
        p
    );
    
    CUDA_CHECK(cudaGetLastError());
}

}
}
