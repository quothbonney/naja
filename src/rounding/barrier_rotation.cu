#include "rounding/barrier_rotation.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "dmatrix.h"
#include "dvector.h"
#include "helper.h"

// cuSOLVER error-checking macro (throws on failure, unlike CUDA_CHECK/CUBLAS_CHECK which only print)
#define CUSOLVER_CHECK(expr)                                                         \
    do {                                                                             \
        cusolverStatus_t _cs = (expr);                                               \
        if (_cs != CUSOLVER_STATUS_SUCCESS) {                                        \
            throw std::runtime_error(                                                \
                std::string("cuSolver error ") + std::to_string(static_cast<int>(_cs)) + \
                " at " __FILE__ ":" + std::to_string(__LINE__));                     \
        }                                                                            \
    } while (0)

namespace {

// Scale rows of a column-major matrix B (m×d) in-place:
//   B[i, j] *= inv_s[i]  for all i in [0,m), j in [0,d)
// Column-major: element [i,j] lives at offset j*m + i → linear index = j*m + i
// So the row of the linear index idx is (idx % m).
__global__ void scale_rows_kernel(double* __restrict__ B,
                                   const double* __restrict__ inv_s,
                                   int m, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int total = m * d;
    for (; idx < total; idx += stride) {
        B[idx] *= inv_s[idx % m];
    }
}

} // anonymous namespace

namespace naja::rounding {

BarrierRotation compute_barrier_rotation(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& x_c,
    bool verbose) {

    const int m = static_cast<int>(A.rows());
    const int d = static_cast<int>(A.cols());

    // ── 1. CPU: slack, tau, clamping  (O(m*d) but cheap vs GPU setup) ─────────
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
        if (slack[i] < tau) { slack[i] = tau; ++n_clamped; }
    }
    Eigen::VectorXd inv_slack = slack.cwiseInverse();

    // ── 2. GPU: B = diag(inv_slack) * A  (row-scale A in place on device) ─────
    naja::gpu::DMatrix<double> B_d(A);                  // m × d, column-major
    naja::gpu::DVector<double> inv_slack_d(inv_slack);  // m

    {
        const int total = m * d;
        const int block = 256;
        const int grid = std::min((total + block - 1) / block, 65535);
        scale_rows_kernel<<<grid, block>>>(B_d.dmat, inv_slack_d.dvec, m, d);
        CUDA_CHECK(cudaGetLastError());
    }

    // ── 3. GPU: H = B^T * B  (d × d symmetric, lower triangle via DSYRK) ──────
    // cublasDsyrk with CUBLAS_OP_T:  C = alpha * B^T * B + beta * C
    //   n=d (output dim), k=m (inner dim), lda=m (leading dim of B)
    cublasHandle_t blas_h;
    CUBLAS_CHECK(cublasCreate(&blas_h));

    naja::gpu::DMatrix<double> H_d(d, d);
    {
        const double alpha = 1.0, beta = 0.0;
        CUBLAS_CHECK(cublasDsyrk(
            blas_h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
            d, m, &alpha, B_d.dmat, m, &beta, H_d.dmat, d));
    }
    cublasDestroy(blas_h);

    // ── 4. Download H (lower triangle), fill upper, add ridge, re-upload ───────
    // H_d is d×d; toHost() copies all d*d elements (lower triangle filled, rest zero).
    Eigen::MatrixXd H_host = H_d.toHost();

    // Mirror lower → upper
    for (int i = 0; i < d; ++i)
        for (int j = i + 1; j < d; ++j)
            H_host(i, j) = H_host(j, i);

    const double trace_H = H_host.trace();
    const double ridge    = 1e-10 * trace_H / d;
    H_host.diagonal().array() += ridge;

    // Re-upload the full symmetric H with ridge
    naja::gpu::DMatrix<double> H_sym_d(H_host);  // d × d

    // ── 5. GPU: eigendecomposition via cuSOLVER DSYEVD ───────────────────────
    //   jobz = CUSOLVER_EIG_MODE_VECTOR → compute eigenvalues + eigenvectors
    //   uplo = CUBLAS_FILL_MODE_LOWER   → reads lower triangle of H_sym_d
    //   On exit: H_sym_d.dmat contains eigenvectors (columns), ascending by eigenvalue
    cusolverDnHandle_t solver_h;
    CUSOLVER_CHECK(cusolverDnCreate(&solver_h));

    // Allocate eigenvalue buffer on device
    double* d_W = nullptr;
    CUDA_CHECK(cudaMalloc(&d_W, static_cast<size_t>(d) * sizeof(double)));

    // Query workspace size
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(
        solver_h, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
        d, H_sym_d.dmat, d, d_W, &lwork));

    double* d_work = nullptr;
    CUDA_CHECK(cudaMalloc(&d_work, static_cast<size_t>(lwork) * sizeof(double)));

    int* d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    // Run eigendecomposition (overwrites H_sym_d with eigenvectors)
    CUSOLVER_CHECK(cusolverDnDsyevd(
        solver_h, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
        d, H_sym_d.dmat, d, d_W, d_work, lwork, d_info));

    // Check convergence
    int h_info = 0;
    CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_work);
    cudaFree(d_info);
    cusolverDnDestroy(solver_h);

    if (h_info != 0) {
        cudaFree(d_W);
        throw std::runtime_error(
            "cusolverDnDsyevd failed: info=" + std::to_string(h_info) +
            " (d=" + std::to_string(d) + ")");
    }

    // Download eigenvalues (ascending)
    Eigen::VectorXd eigenvalues(d);
    CUDA_CHECK(cudaMemcpy(eigenvalues.data(), d_W,
                          static_cast<size_t>(d) * sizeof(double),
                          cudaMemcpyDeviceToHost));
    cudaFree(d_W);

    // Download eigenvectors (columns of H_sym_d after dsyevd)
    Eigen::MatrixXd Q = H_sym_d.toHost();  // d × d, columns = eigenvectors

    // ── 6. Optional verbose diagnostics ──────────────────────────────────────
    if (verbose) {
        // Diagonalization sanity check: ||off-diag(Q'HQ)|| / ||Q'HQ||
        Eigen::MatrixXd D = Q.transpose() * H_host * Q;
        double off_diag_sq = 0.0;
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j)
                if (i != j) off_diag_sq += D(i, j) * D(i, j);
        double total_sq = D.squaredNorm();
        double diag_quality = (total_sq > 0.0) ? std::sqrt(off_diag_sq / total_sq) : 0.0;

        double eig_min = eigenvalues[0];
        double eig_max = eigenvalues[d - 1];
        double eig_med = eigenvalues[d / 2];

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

    BarrierRotation result;
    result.Q           = std::move(Q);
    result.eigenvalues = std::move(eigenvalues);
    return result;
}

} // namespace naja::rounding
