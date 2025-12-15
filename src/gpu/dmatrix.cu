#include "dmatrix.h"
#include "helper.h"

#include <algorithm>

namespace naja {
    namespace gpu {

        template<typename Real>
        __global__ void broadcastColumnsKernel(const Real* __restrict__ src, Real* __restrict__ dst, int rows, int cols_to_fill) {
            int row = blockIdx.x * blockDim.x + threadIdx.x;
            int col = blockIdx.y * blockDim.y + threadIdx.y;
            if (row < rows && col < cols_to_fill) {
                dst[IDX2C(row, col, rows)] = src[row];
            }
        }

        template<typename Real>
        inline void broadcastColumns(const Real* src, Real* dst, int rows, int cols_to_fill) {
            if (cols_to_fill <= 0 || rows <= 0) {
                return;
            }
            dim3 block(32, 8);
            dim3 grid((rows + block.x - 1) / block.x,
                      (cols_to_fill + block.y - 1) / block.y);
            broadcastColumnsKernel<<<grid, block>>>(src, dst, rows, cols_to_fill);
            CUDA_CHECK(cudaGetLastError());
        }

        /*************************************/
        /*********** Constructors ************/
        /************************************/

        // Construct from Eigen
        template<typename Real>
        DMatrix<Real>::DMatrix(const HMatrix<Real>& m) : rows(m.rows()), cols(m.cols()), ownsData(true) {
            size_t size = rows * cols * sizeof(Real);
            CUDA_CHECK(cudaMalloc(&dmat, size));
            CUDA_CHECK(cudaMemcpy(dmat, m.data(), rows * cols * sizeof(Real), cudaMemcpyHostToDevice));
        }

        // Construct from input dimensions
        template<typename Real>
        DMatrix<Real>::DMatrix(int rows_, int cols_) : rows(rows_), cols(cols_), ownsData(true) {
            size_t size = rows * cols * sizeof(Real);
            CUDA_CHECK(cudaMalloc(&dmat, size));
        }

        // Construct from vector and broadcast
        template<typename Real>
        DMatrix<Real>::DMatrix(const HVector<Real>& v, int cols) : rows(v.size()), cols(cols), ownsData(true) {
            size_t size = rows * cols * sizeof(Real);
            CUDA_CHECK(cudaMalloc(&dmat, size));
            if (cols > 0) {
                CUDA_CHECK(cudaMemcpy(dmat, v.data(), rows * sizeof(Real), cudaMemcpyHostToDevice));
                broadcastColumns(dmat, dmat, rows, cols);
            }
        }

        // Construct from device vector and broadcast
        template<typename Real>
        DMatrix<Real>::DMatrix(const DVector<Real>& v, int cols) : rows(v.len), cols(cols), ownsData(true) {
            size_t size = rows * cols * sizeof(Real);
            CUDA_CHECK(cudaMalloc(&dmat, size));
            if (cols > 0) {
                broadcastColumns(v.dvec, dmat, rows, cols);
            }
        }

        // Construct from vector with a given number of cols, and broadcast to 'z' cols
        template<typename Real>
        DMatrix<Real>::DMatrix(const HVector<Real>& v, int cols, int z)  : rows(v.size()), cols(cols), ownsData(true) {
            size_t size = rows * cols * sizeof(Real);
            CUDA_CHECK(cudaMalloc(&dmat, size));
            int cols_to_fill = std::min(z, cols);
            if (cols_to_fill > 0) {
                CUDA_CHECK(cudaMemcpy(dmat, v.data(), rows * sizeof(Real), cudaMemcpyHostToDevice));
                broadcastColumns(dmat, dmat, rows, cols_to_fill);
            }
        }

        // Construct subMatrix view
        template<typename Real>
        DMatrix<Real>::DMatrix(Real *dmat_, int rows_, int cols_) : dmat(dmat_), rows(rows_), cols(cols_), ownsData(false) {}

        // Copy constructor
        template<typename Real>
        DMatrix<Real>::DMatrix(const DMatrix &other) : rows(other.rows), cols(other.cols), ownsData(true) {
            size_t size = rows * cols * sizeof(Real);
            CUDA_CHECK(cudaMalloc(&dmat, size));
            CUDA_CHECK(cudaMemcpy(dmat, other.dmat, rows * cols * sizeof(Real), cudaMemcpyDeviceToDevice));
        }

        // Move constructor
        template<typename Real>
        DMatrix<Real>::DMatrix(DMatrix &&other) noexcept : dmat(other.dmat), rows(other.rows), cols(other.cols), ownsData(other.ownsData) {
            other.dmat = nullptr;
            other.ownsData = false;
        }

        // Destructor
        template<typename Real>
        DMatrix<Real>::~DMatrix() {
            if(ownsData){ cudaFree(dmat); dmat=nullptr;}
        }

        /*************************************/
        /************* Methods****************/
        /************************************/

        template<typename Real>
        HMatrix<Real> DMatrix<Real>::toHost() const {
            HMatrix<Real> m(rows, cols);
            CUDA_CHECK(cudaMemcpy(m.data(), dmat, rows * cols * sizeof(Real), cudaMemcpyDeviceToHost));
            return m;
        }

        template<typename Real>
        DMatrix<Real> DMatrix<Real>::subMatrix(int rowStart, int colStart, int rowEnd, int colEnd) const {
            int subRows = rowEnd - rowStart;
            int subCols = colEnd - colStart;
            Real *subMat = dmat + IDX2C(rowStart, colStart, rows);
            return DMatrix<Real>(subMat, subRows, subCols);
        }


        // Instantiation
        template class DMatrix<double>;
    }
}
