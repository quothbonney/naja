#ifdef NDEBUG
#undef NDEBUG
#endif
#include "gpu/helper.h"
#include <cassert>
#include <cstdint>

__global__ void index_probe(std::uint64_t* result) {
    result[0] = IDX2C(127, 4000000, 800);
    const std::uint64_t samples = 20000000, thinning = 500;
    result[1] = samples * thinning;
}

int main() {
    bool cuda_threw = false, blas_threw = false, solver_threw = false;
    try { CUDA_CHECK(cudaErrorInvalidValue); }
    catch (const std::runtime_error&) { cuda_threw = true; }
    try { CUBLAS_CHECK(CUBLAS_STATUS_INVALID_VALUE); }
    catch (const std::runtime_error&) { blas_threw = true; }
    try { CUSOLVER_CHECK(CUSOLVER_STATUS_INVALID_VALUE); }
    catch (const std::runtime_error&) { solver_threw = true; }
    assert(cuda_threw && blas_threw && solver_threw);
    std::uint64_t* device = nullptr;
    CUDA_CHECK(cudaMalloc(&device, 2 * sizeof(std::uint64_t)));
    index_probe<<<1,1>>>(device);
    CUDA_CHECK(cudaGetLastError());
    std::uint64_t host[2] = {};
    CUDA_CHECK(cudaMemcpy(host, device, sizeof(host), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device));
    assert(host[0] == UINT64_C(3200000127));
    assert(host[1] == UINT64_C(10000000000));
    return 0;
}
