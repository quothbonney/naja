#pragma once
#include "dmatrix.h"
#include <Eigen/Dense>


namespace naja {
    namespace gpu {
        
        struct CudaEvent{
            cudaEvent_t e;
            CudaEvent(){CUDA_CHECK(cudaEventCreate(&e));}
            CudaEvent(unsigned int flags){CUDA_CHECK(cudaEventCreateWithFlags(&e, flags));}
            ~CudaEvent(){CUDA_CHECK_NOEXCEPT(cudaEventDestroy(e));}
            operator cudaEvent_t() const { return e; }
        };

        struct CudaStream{
            cudaStream_t s;
            CudaStream(){CUDA_CHECK(cudaStreamCreate(&s));}
            CudaStream(unsigned int flags){CUDA_CHECK(cudaStreamCreateWithFlags(&s, flags));}
            ~CudaStream(){CUDA_CHECK_NOEXCEPT(cudaStreamDestroy(s));}
            operator cudaStream_t() const { return s; }
        };

        template <typename T>
        struct DevPtr{
            T* ptr;
            DevPtr(){   CUDA_CHECK(cudaMalloc(&ptr, sizeof(T)));    }
            DevPtr(T val){  CUDA_CHECK(cudaMalloc(&ptr, sizeof(T))); CUDA_CHECK(cudaMemcpy(ptr, &val, sizeof(T), cudaMemcpyHostToDevice)); }
            ~DevPtr(){  CUDA_CHECK_NOEXCEPT(cudaFree(ptr));   }
            T toHost() const { T val; CUDA_CHECK(cudaMemcpy(&val, ptr, sizeof(T), cudaMemcpyDeviceToHost)); return val; }
            void set(T val) { CUDA_CHECK(cudaMemcpy(ptr, &val, sizeof(T), cudaMemcpyHostToDevice)); }
            operator T*() const { return ptr; }
        };

        template <typename Func>
        inline std::pair<int, int> getOptimalLaunchConfig(Func kernelFunc) {
            int maxBlocksPerSM = 0;
            int optimalThreadsPerBlock = 0;
            CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&maxBlocksPerSM, &optimalThreadsPerBlock, kernelFunc, 0, 0));
            return {optimalThreadsPerBlock, maxBlocksPerSM};
        }
    }
}