#include "dmatrix.h"
#include "helper.h"

#include <algorithm>
#include <thread>
#include <vector>
#include <unistd.h>

namespace naja {
    namespace gpu {

        template<typename Real>
        __global__ void broadcastColumnsKernel(const Real* __restrict__ src, Real* __restrict__ dst, int rows, int cols_to_fill) {
            const std::size_t total = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols_to_fill);
            const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
            for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                 idx < total; idx += stride) dst[idx] = src[idx % static_cast<std::size_t>(rows)];
        }

        template<typename Real>
        inline void broadcastColumns(const Real* src, Real* dst, int rows, int cols_to_fill) {
            if (cols_to_fill <= 0 || rows <= 0) return;
            const std::size_t total = naja::checked_size_mul(rows, cols_to_fill, "broadcast columns");
            const int block = 256;
            const int grid = static_cast<int>(std::min<std::size_t>((total - 1) / block + 1, 65535));
            broadcastColumnsKernel<<<grid, block>>>(src, dst, rows, cols_to_fill);
            CUDA_CHECK(cudaGetLastError());
        }

        /*************************************/
        /*********** Constructors ************/
        /************************************/

        // Construct from Eigen
        template<typename Real>
        DMatrix<Real>::DMatrix(const HMatrix<Real>& m) : rows(naja::checked_int_dimension(m.rows(), "device matrix rows")), cols(naja::checked_int_dimension(m.cols(), "device matrix columns")), ownsData(true) {
            size_t size = naja::checked_matrix_bytes<Real>(rows, cols, "device matrix");
            CUDA_CHECK(cudaMalloc(&dmat, size));
            CUDA_CHECK(cudaMemcpy(dmat, m.data(), naja::checked_matrix_bytes<Real>(rows, cols, "device matrix"), cudaMemcpyHostToDevice));
        }

        // Construct from input dimensions
        template<typename Real>
        DMatrix<Real>::DMatrix(int rows_, int cols_) : rows(rows_), cols(cols_), ownsData(true) {
            size_t size = naja::checked_matrix_bytes<Real>(rows, cols, "device matrix");
            CUDA_CHECK(cudaMalloc(&dmat, size));
        }

        // Construct from vector and broadcast
        template<typename Real>
        DMatrix<Real>::DMatrix(const HVector<Real>& v, int cols) : rows(naja::checked_int_dimension(v.size(), "device matrix rows")), cols(cols), ownsData(true) {
            size_t size = naja::checked_matrix_bytes<Real>(rows, cols, "device matrix");
            CUDA_CHECK(cudaMalloc(&dmat, size));
            if (cols > 0) {
                CUDA_CHECK(cudaMemcpy(dmat, v.data(), naja::checked_matrix_bytes<Real>(rows, 1, "device matrix column"), cudaMemcpyHostToDevice));
                broadcastColumns(dmat, dmat, rows, cols);
            }
        }

        // Construct from device vector and broadcast
        template<typename Real>
        DMatrix<Real>::DMatrix(const DVector<Real>& v, int cols) : rows(v.len), cols(cols), ownsData(true) {
            size_t size = naja::checked_matrix_bytes<Real>(rows, cols, "device matrix");
            CUDA_CHECK(cudaMalloc(&dmat, size));
            if (cols > 0) {
                broadcastColumns(v.dvec, dmat, rows, cols);
            }
        }

        // Construct from vector with a given number of cols, and broadcast to 'z' cols
        template<typename Real>
        DMatrix<Real>::DMatrix(const HVector<Real>& v, int cols, int z)  : rows(naja::checked_int_dimension(v.size(), "device matrix rows")), cols(cols), ownsData(true) {
            size_t size = naja::checked_matrix_bytes<Real>(rows, cols, "device matrix");
            CUDA_CHECK(cudaMalloc(&dmat, size));
            int cols_to_fill = std::min(z, cols);
            if (cols_to_fill > 0) {
                CUDA_CHECK(cudaMemcpy(dmat, v.data(), naja::checked_matrix_bytes<Real>(rows, 1, "device matrix column"), cudaMemcpyHostToDevice));
                broadcastColumns(dmat, dmat, rows, cols_to_fill);
            }
        }

        // Construct subMatrix view
        template<typename Real>
        DMatrix<Real>::DMatrix(Real *dmat_, int rows_, int cols_) : dmat(dmat_), rows(rows_), cols(cols_), ownsData(false) {
            (void)naja::checked_matrix_bytes<Real>(rows, cols, "device matrix view");
        }

        // Copy constructor
        template<typename Real>
        DMatrix<Real>::DMatrix(const DMatrix &other) : rows(other.rows), cols(other.cols), ownsData(true) {
            size_t size = naja::checked_matrix_bytes<Real>(rows, cols, "device matrix");
            CUDA_CHECK(cudaMalloc(&dmat, size));
            CUDA_CHECK(cudaMemcpy(dmat, other.dmat, naja::checked_matrix_bytes<Real>(rows, cols, "device matrix"), cudaMemcpyDeviceToDevice));
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
            if(ownsData && dmat){ CUDA_CHECK_NOEXCEPT(cudaFree(dmat)); dmat=nullptr;}
        }

        /*************************************/
        /************* Methods****************/
        /************************************/

        template<typename Real>
        HMatrix<Real> DMatrix<Real>::toHost(int prefault_threads) const {
            HMatrix<Real> m(rows, cols);
            const std::size_t bytes = naja::checked_matrix_bytes<Real>(rows, cols, "device matrix");
            if (prefault_threads > 0 && bytes > 0) {
                const long system_page_size = sysconf(_SC_PAGESIZE);
                const std::size_t page_bytes = system_page_size > 0 ? static_cast<std::size_t>(system_page_size) : 4096;
                const std::size_t pages = 1 + (bytes - 1) / page_bytes;
                const std::size_t count = std::min(static_cast<std::size_t>(prefault_threads), pages);
                volatile unsigned char* target = reinterpret_cast<volatile unsigned char*>(m.data());
                std::vector<std::thread> workers;
                workers.reserve(count);
                try {
                    for (std::size_t tid = 0; tid < count; ++tid) {
                        const std::size_t first = tid * pages / count;
                        const std::size_t last = (tid + 1) * pages / count;
                        workers.emplace_back([target, first, last, page_bytes]() {
                            for (std::size_t page = first; page < last; ++page) target[page * page_bytes] = 0;
                        });
                    }
                } catch (...) {
                    for (auto& worker : workers) worker.join();
                    throw;
                }
                for (auto& worker : workers) worker.join();
                target[bytes - 1] = 0; // Include the tail of a partial final page.
            }
            CUDA_CHECK(cudaMemcpy(m.data(), dmat, bytes, cudaMemcpyDeviceToHost));
            return m;
        }

        template<typename Real>
        DMatrix<Real> DMatrix<Real>::subMatrix(int rowStart, int colStart, int rowEnd, int colEnd) const {
            if (rowStart < 0 || colStart < 0 || rowEnd < rowStart || colEnd < colStart || rowEnd > rows || colEnd > cols)
                throw std::out_of_range("device submatrix bounds");
            int subRows = rowEnd - rowStart;
            int subCols = colEnd - colStart;
            Real *subMat = (subRows == 0 || subCols == 0) ? dmat : dmat + IDX2C(rowStart, colStart, rows);
            return DMatrix<Real>(subMat, subRows, subCols);
        }


        // Instantiation
        template class DMatrix<double>;
    }
}
