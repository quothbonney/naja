#include "backmap.cuh"

#include <stdexcept>
#include <algorithm>

namespace naja::gpu {

template <typename Real>
__global__ void add_shift_cols_kernel(Real * __restrict__ data,
                                      const Real * __restrict__ shift,
                                      int rows,
                                      int cols) {
    const std::size_t total = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += stride) data[idx] += shift[idx % static_cast<std::size_t>(rows)];
}

template <typename Real>
void backmap_into(const DMatrix<Real> &transformation,
                  const DVector<Real> &shift,
                  const DMatrix<Real> &samples,
                  DMatrix<Real> &output,
                  CUBLASHandle &handle,
                  cudaStream_t stream) {
    if (transformation.cols != samples.rows) {
        throw std::invalid_argument("backmap_into: transformation shape mismatch with samples");
    }
    if (shift.len != transformation.rows) {
        throw std::invalid_argument("backmap_into: shift length mismatch with transformation rows");
    }
    if (output.rows != transformation.rows || output.cols != samples.cols) {
        throw std::invalid_argument("backmap_into: output shape mismatch");
    }

    CUBLAS_CHECK(cublasSetStream(handle, stream));
    Real alpha = static_cast<Real>(1.0);
    Real beta = static_cast<Real>(0.0);
    handle.GeMM(transformation, samples, output, alpha, beta);

    const std::size_t total = naja::checked_size_mul(output.rows, output.cols, "backmap output");
    if (total == 0) return;
    const int block = 256;
    const int grid = static_cast<int>(std::min<std::size_t>((total - 1) / block + 1, 65535));

    add_shift_cols_kernel<<<grid, block, 0, stream>>>(output.dmat, shift.dvec, output.rows, output.cols);
    CUDA_CHECK(cudaGetLastError());
}

template <typename Real>
DMatrix<Real> backmap(const DMatrix<Real> &transformation,
                      const DVector<Real> &shift,
                      const DMatrix<Real> &samples,
                      CUBLASHandle &handle,
                      cudaStream_t stream) {
    DMatrix<Real> output(transformation.rows, samples.cols);
    backmap_into(transformation, shift, samples, output, handle, stream);
    return output;
}

template void backmap_into<double>(const DMatrix<double> &, const DVector<double> &, const DMatrix<double> &, DMatrix<double> &, CUBLASHandle &, cudaStream_t);
template DMatrix<double> backmap<double>(const DMatrix<double> &, const DVector<double> &, const DMatrix<double> &, CUBLASHandle &, cudaStream_t);

} // namespace naja::gpu

