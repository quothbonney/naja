    #pragma once
#include <cstddef>


    namespace naja {
        namespace gpu {
            
            template<typename Real>
            __global__ void initializeKernel(Real* data, int len, Real value) {
                const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                if (idx < static_cast<std::size_t>(len)) {
                    data[idx] = value;
                }
            }
        }
    }        