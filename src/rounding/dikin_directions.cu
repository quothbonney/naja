#include "rounding/dikin_directions.h"

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "dmatrix.h"
#include "dvector.h"
#include "helper.h"

#define CUSOLVER_CHECK(expr)                                                         \
    do {                                                                             \
        cusolverStatus_t _cs = (expr);                                               \
        if (_cs != CUSOLVER_STATUS_SUCCESS) {                                        \
            throw std::runtime_error(                                                \
                std::string("cuSolver error ") + std::to_string(static_cast<int>(_cs)) + \
                " at " __FILE__ ":" + std::to_string(__LINE__));                     \
        }                                                                            \
    } while (0)

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

    // ── CPU: build delta-Hessian from extra rows ───────────────────────────────
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

    // Build delta-Hessian: H = sum_{tight extra rows} (1/delta_i^2) n_i n_i'
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

    // ── GPU: eigendecompose H via cuSOLVER DSYEVD ─────────────────────────────
    naja::gpu::DMatrix<double> H_d(H);  // d × d, column-major (Eigen is column-major)

    cusolverDnHandle_t solver_h;
    CUSOLVER_CHECK(cusolverDnCreate(&solver_h));

    double* d_W = nullptr;
    CUDA_CHECK(cudaMalloc(&d_W, static_cast<size_t>(d) * sizeof(double)));

    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(
        solver_h, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
        d, H_d.dmat, d, d_W, &lwork));

    double* d_work = nullptr;
    CUDA_CHECK(cudaMalloc(&d_work, static_cast<size_t>(lwork) * sizeof(double)));

    int* d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    CUSOLVER_CHECK(cusolverDnDsyevd(
        solver_h, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
        d, H_d.dmat, d, d_W, d_work, lwork, d_info));

    int h_info = 0;
    CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_work);
    cudaFree(d_info);
    cusolverDnDestroy(solver_h);

    if (h_info != 0) {
        cudaFree(d_W);
        throw std::runtime_error(
            "cusolverDnDsyevd (dikin) failed: info=" + std::to_string(h_info));
    }

    // Download eigenvalues (ascending) and eigenvectors
    Eigen::VectorXd eigenvalues(d);
    CUDA_CHECK(cudaMemcpy(eigenvalues.data(), d_W,
                          static_cast<size_t>(d) * sizeof(double),
                          cudaMemcpyDeviceToHost));
    cudaFree(d_W);

    Eigen::MatrixXd eigenvectors = H_d.toHost();  // d × d, columns = eigenvectors

    // ── Select top-k escape directions ────────────────────────────────────────
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

    // Extract top-k directions (highest eigenvalue columns = last columns)
    result.V.resize(d, k);
    for (int j = 0; j < k; ++j) {
        result.V.col(j) = eigenvectors.col(d - 1 - j);
    }
    result.Av = A * result.V;  // (m, k)
    result.k  = k;

    return result;
}

} // namespace naja::rounding
