#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <string>

// CUDA API error checking
#define CUDA_CHECK(cuda_error) \
    do { \
        cudaError_t err = cuda_error; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        } \
    } while (0)

// CUBLAS API error checking
#define CUBLAS_CHECK(cublas_error) \
    do { \
        cublasStatus_t err = cublas_error; \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            throw std::runtime_error(std::string("cuBLAS error: ") + cublasGetStatusString(err) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        } \
    } while (0)

// Destructors/move assignment cannot throw during stack unwinding. A cleanup
// failure is fatal rather than allowing an apparently successful CLI exit.
#define CUDA_CHECK_NOEXCEPT(expr) do { \
    cudaError_t err_ = (expr); \
    if (err_ != cudaSuccess) { \
        std::cerr << "CUDA cleanup error: " << cudaGetErrorString(err_) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::abort(); \
    } \
} while (0)
#define CUBLAS_CHECK_NOEXCEPT(expr) do { \
    cublasStatus_t err_ = (expr); \
    if (err_ != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS cleanup error: " << cublasGetStatusString(err_) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::abort(); \
    } \
} while (0)

#define CUSOLVER_CHECK(expr) do { \
    cusolverStatus_t err_ = (expr); \
    if (err_ != CUSOLVER_STATUS_SUCCESS) \
        throw std::runtime_error(std::string("cuSOLVER error: ") + std::to_string(static_cast<int>(err_)) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
} while (0)


// CURAND API error checking
#define CURAND_CHECK(err)                                                      \
  do {                                                                         \
    curandStatus_t err_ = (err);                                               \
    if (err_ != CURAND_STATUS_SUCCESS) {                                       \
      std::printf("curand error %d at %s:%d\n", err_, __FILE__, __LINE__);     \
      throw std::runtime_error("curand error");                                \
    }                                                                          \
  } while (0)
