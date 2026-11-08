#pragma once
#include "helper.h"

namespace naja {
    namespace gpu {
        /**
         * @brief RAII Wrapper class for managing GPU memory allocation of PRNG state arrays.
         *
         * @tparam PRNGenerator The type representing the PRNG state (e.g., curandState).
         */
        template<typename PRNGenerator>
        struct PRNGState{
            PRNGenerator* states;
            public:
                PRNGState(int nThreads){ CUDA_CHECK(cudaMalloc((void **)&states, naja::checked_matrix_bytes<PRNGenerator>(nThreads, 1, "PRNG states"))); }
                ~PRNGState(){ CUDA_CHECK_NOEXCEPT(cudaFree(states)); }
        };

    }
}