#ifdef NDEBUG
#undef NDEBUG
#endif
#include "gpu/dmatrix.h"
#include "gpu/helper.h"
#include "npy.h"
#include <cassert>
#include <fstream>
#include <iostream>

__global__ void sentinels(double* data, int rows, int cols) {
    data[0] = 42.0;
    data[IDX2C(17, cols - 1, rows)] = 3.25;
    data[IDX2C(rows - 1, cols - 1, rows)] = -7.0;
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    try {
        constexpr int rows = 512, cols = 4194305;
        const std::size_t bytes = naja::checked_matrix_bytes<double>(rows, cols, "positive large-array smoke");
        std::size_t free = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        if (free < bytes + (UINT64_C(2) << 30)) {
            std::ofstream(argv[1]) << "{\"status\":\"skipped\",\"reason\":\"insufficient GPU memory\"}\n";
            return 77;
        }
        naja::gpu::DMatrix<double> matrix(rows, cols);
        sentinels<<<1,1>>>(matrix.dmat, rows, cols);
        CUDA_CHECK(cudaGetLastError());
        const Eigen::MatrixXd host = matrix.toHost(0);
        assert(host(0,0) == 42.0);
        assert(host(17,cols-1) == 3.25);
        assert(host(rows-1,cols-1) == -7.0);
        npy::save_f64_samples("/dev/null", host);
        std::ofstream out(argv[1]);
        out << "{\"status\":\"passed\",\"rows\":" << rows << ",\"cols\":" << cols
            << ",\"elements\":" << static_cast<std::uint64_t>(rows)*cols
            << ",\"bytes\":" << bytes
            << ",\"largest_checked_element_offset\":" << IDX2C(rows-1,cols-1,rows)
            << ",\"full_device_to_host_copy\":true,\"float64_stream_sink\":\"/dev/null\",\"persistent_payload_written\":false}\n";
        std::cout << "Positive wide-allocation/copy/export smoke passed, " << bytes << " bytes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
