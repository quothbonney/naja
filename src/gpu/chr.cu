#include "chr.cuh"
#include "gpusamplers.h"
#include "backmap.cuh"
#include "pinned_host.hpp"

#include <algorithm>
#include <functional>
#include <stdexcept>

namespace naja {
    namespace gpu {

        DMatrix<double> CoordinateHitAndRun(    DMatrix<double>& A_d,
                                        DVector<double>& b_d,
                                        DMatrix<double>& X_d,
                                        int nspc,
                                        int thinning,
                                        int nchains,
                                        int tpb_ss)
        {
            // TODO: Implement logic for launch config when tpb_ss are not provided
            int N = A_d.cols;

            DMatrix<double> samples_d(N, nchains * nspc), slack_d(b_d, nchains);

            // Init cuBLAS handle
            CUBLASHandle handle;

            // Init slack: slack = b - A*x0
            handle.GeMM(A_d, X_d, slack_d, -1.0, 1.0);

            // Init PRNG
            PRNGState<curandStateMRG32k3a> gen(nchains);
            std::pair<int, int> launchConfig = getOptimalLaunchConfig(initPrngStatesCHR<curandStateMRG32k3a>);
            int threadsPerBlock = launchConfig.first;
            int blocksPerGrid = (nchains + threadsPerBlock - 1) / threadsPerBlock;
            initPrngStatesCHR<<<blocksPerGrid, threadsPerBlock>>>(gen.states, 0, nchains);
            CUDA_CHECK(cudaGetLastError());

            int threads_ss = (tpb_ss > 0) ? tpb_ss : 128;
            int blocks_ss = nchains;

            // Launch CHR kernel
            launchChrKernel(A_d, slack_d, X_d, samples_d, gen.states, nspc, thinning, threads_ss, blocks_ss, false, 0);


            return samples_d;
        }

        DMatrix<double> CoordinateHitAndRunBackmap(    DMatrix<double>& A_d,
                                                       DVector<double>& b_d,
                                                       DMatrix<double>& X_d,
                                                       const DMatrix<double>& transformation_d,
                                                       const DVector<double>& shift_d,
                                                       int nspc,
                                                       int thinning,
                                                       int nchains,
                                                       int tpb_ss)
        {
            DMatrix<double> samples_d = CoordinateHitAndRun(A_d, b_d, X_d, nspc, thinning, nchains, tpb_ss);

            CUBLASHandle handle;
            DMatrix<double> result_d(transformation_d.rows, samples_d.cols);
            backmap_into<double>(transformation_d, shift_d, samples_d, result_d, handle);

            return result_d;
        }

        void CoordinateHitAndRunStreamed(       DMatrix<double>& A_d,
                                                DVector<double>& b_d,
                                                DMatrix<double>& X_d,
                                                int nspc_total,
                                                int chunk_nspc,
                                                int thinning,
                                                int nchains,
                                                int tpb_ss,
                                                const std::function<void(const double*, int, int)>& host_sink)
        {
            if (nchains <= 0) {
                throw std::invalid_argument("CoordinateHitAndRunStreamed: nchains must be positive");
            }
            if (chunk_nspc <= 0) {
                throw std::invalid_argument("CoordinateHitAndRunStreamed: chunk_nspc must be positive");
            }
            if (nspc_total <= 0) {
                return;
            }
            if (!host_sink) {
                throw std::invalid_argument("CoordinateHitAndRunStreamed: host_sink must be callable");
            }

            const int N = A_d.cols;
            if (N <= 0) {
                return;
            }

            const int max_cols_per_chunk = nchains * chunk_nspc;
            if (max_cols_per_chunk <= 0) {
                throw std::invalid_argument("CoordinateHitAndRunStreamed: invalid chunk sizing");
            }

            const size_t max_chunk_bytes = static_cast<size_t>(N) * static_cast<size_t>(max_cols_per_chunk) * sizeof(double);

            CUBLASHandle handle;
            CudaStream compute_stream;
            CudaStream copy_stream;
            CUBLAS_CHECK(cublasSetStream(handle, compute_stream));

            DMatrix<double> slack_d(b_d, nchains);
            handle.GeMM(A_d, X_d, slack_d, -1.0, 1.0);

            PRNGState<curandStateMRG32k3a> gen(nchains);
            auto launchConfig = getOptimalLaunchConfig(initPrngStatesCHR<curandStateMRG32k3a>);
            int threadsPerBlock = launchConfig.first;
            int blocksPerGrid = (nchains + threadsPerBlock - 1) / threadsPerBlock;
            initPrngStatesCHR<<<blocksPerGrid, threadsPerBlock, 0, compute_stream>>>(gen.states, 0, nchains);
            CUDA_CHECK(cudaGetLastError());

            DMatrix<double> dev_chunk[2] = {
                DMatrix<double>(N, max_cols_per_chunk),
                DMatrix<double>(N, max_cols_per_chunk)
            };
            PinnedHostBuffer host_chunk[2] = {
                PinnedHostBuffer(max_chunk_bytes),
                PinnedHostBuffer(max_chunk_bytes)
            };
            CudaEvent compute_done[2];
            CudaEvent copy_done[2];
            int cols_actual[2] = {0, 0};
            size_t bytes_actual[2] = {0, 0};

            const int n_chunks = (nspc_total + chunk_nspc - 1) / chunk_nspc;
            const int tpb = (tpb_ss > 0) ? tpb_ss : 128;
            const int blocks_ss = nchains;

            for (int c = 0; c < n_chunks; ++c) {
                const int cur = c & 1;
                const int prev = cur ^ 1;
                const int samples_remaining = nspc_total - c * chunk_nspc;
                const int nspc_this = std::min(chunk_nspc, samples_remaining);
                const int cols_this = nchains * nspc_this;
                const size_t bytes_this = static_cast<size_t>(N) * static_cast<size_t>(cols_this) * sizeof(double);

                cols_actual[cur] = cols_this;
                bytes_actual[cur] = bytes_this;

                launchChrKernel(A_d, slack_d, X_d, dev_chunk[cur], gen.states,
                                nspc_this, thinning, tpb, blocks_ss,
                                true, compute_stream);
                CUDA_CHECK(cudaEventRecord(compute_done[cur], compute_stream));

                if (c > 0) {
                    CUDA_CHECK(cudaStreamWaitEvent(copy_stream, compute_done[prev], 0));
                    CUDA_CHECK(cudaMemcpyAsync(host_chunk[prev].data(),
                                               dev_chunk[prev].dmat,
                                               bytes_actual[prev],
                                               cudaMemcpyDeviceToHost,
                                               copy_stream));
                    CUDA_CHECK(cudaEventRecord(copy_done[prev], copy_stream));
                    CUDA_CHECK(cudaEventSynchronize(copy_done[prev]));
                    if (cols_actual[prev] > 0) {
                        host_sink(static_cast<const double*>(host_chunk[prev].data()), N, cols_actual[prev]);
                    }
                }
            }

            const int last = (n_chunks - 1) & 1;
            CUDA_CHECK(cudaStreamWaitEvent(copy_stream, compute_done[last], 0));
            CUDA_CHECK(cudaMemcpyAsync(host_chunk[last].data(),
                                       dev_chunk[last].dmat,
                                       bytes_actual[last],
                                       cudaMemcpyDeviceToHost,
                                       copy_stream));
            CUDA_CHECK(cudaStreamSynchronize(copy_stream));
            if (cols_actual[last] > 0) {
                host_sink(static_cast<const double*>(host_chunk[last].data()), N, cols_actual[last]);
            }
        }

        template <typename Real, typename PRNGenerator, int ThreadsPerBlock>
        __global__ void chrKernel(Real* A, Real* slack, Real* X0, Real* samples, int rows, int cols, PRNGenerator* global_states, int samples_per_chain, int thinning, int nchains, int persist_state){

            int tid = threadIdx.x;
            int bid = blockIdx.x;
            int threadstride = blockDim.x;

            // Shared memory for x
            extern __shared__ Real x_s[];
            // Copy x0 to shared memory
            for (int i = tid; i < cols; i += threadstride){
                x_s[i] = X0[IDX2C(i, bid, cols)];
            }

            using BlockReduce = cub::BlockReduce<Real, ThreadsPerBlock>;
            __shared__ typename BlockReduce::TempStorage temp_storage_max;
            __shared__ typename BlockReduce::TempStorage temp_storage_min;
            __shared__ Real alpha;

            for (int t = 0; t < samples_per_chain * thinning; ++t){
                
                bool save = t % thinning == 0;
                int colOffset = static_cast<int>(t / thinning) * nchains;
                int e = t % cols;

                Real partial_max = cuda::std::numeric_limits<Real>::lowest();
                Real partial_min = cuda::std::numeric_limits<Real>::max();
                for (int i = tid; i < rows; i += threadstride){

                    // Get slack and projected direction
                    Real s = slack[IDX2C(i, bid, rows)];
                    Real ae = A[IDX2C(i, e, rows)];
                    
                    // Get step size bounds
                    Real inv_dist = ae / s;
                    inv_dist = isnan(inv_dist) || isinf(inv_dist) ? Real(0.0) : inv_dist;
                    partial_max = fmax(partial_max, inv_dist);
                    partial_min = fmin(partial_min, inv_dist);
                }
                Real aggregate_max = BlockReduce(temp_storage_max).Reduce(partial_max, cub::Max());
                Real aggregate_min = BlockReduce(temp_storage_min).Reduce(partial_min, cub::Min());
                if(tid==0){
                    PRNGenerator localState = global_states[bid];
                    Real usample = curand_uniform_double(&localState);
                    alpha = (1/aggregate_min) + usample * ((1/aggregate_max) - (1/aggregate_min));
                    x_s[e] += alpha;
                    global_states[bid] = localState;
                }
                __syncthreads();


                // save x
                if(save){
                    for (int i = tid; i < cols; i += threadstride)
                        samples[IDX2C(i, bid + colOffset, cols)] = x_s[i];
                }
                
                // Update slack
                for (int i = tid; i < rows; i += threadstride){
                    slack[IDX2C(i, bid, rows)] -= alpha * A[IDX2C(i, e, rows)];
                }
            }

            if (persist_state) {
                for (int i = tid; i < cols; i += threadstride) {
                    X0[IDX2C(i, bid, cols)] = x_s[i];
                }
            }
        }
        template __global__ void chrKernel<double, curandState, 32>(double*, double*, double*, double*, int, int, curandState*, int, int, int, int);
        template __global__ void chrKernel<double, curandState, 64>(double*, double*, double*, double*, int, int, curandState*, int, int, int, int);
        template __global__ void chrKernel<double, curandState, 128>(double*, double*, double*, double*, int, int, curandState*, int, int, int, int);
        template __global__ void chrKernel<double, curandState, 256>(double*, double*, double*, double*, int, int, curandState*, int, int, int, int);
        template __global__ void chrKernel<double, curandState, 512>(double*, double*, double*, double*, int, int, curandState*, int, int, int, int);
        template __global__ void chrKernel<double, curandState, 1024>(double*, double*, double*, double*, int, int, curandState*, int, int, int, int);

        template <typename Real, typename PRNGenerator>
        void launchChrKernel(DMatrix<Real>& A, DMatrix<Real>& slack, DMatrix<Real>& x0, DMatrix<Real>& samples, PRNGenerator* global_states, int samples_per_chain, int thinning, int threads_per_block, int blocks_per_grid, bool persist_state, cudaStream_t stream){
            int rows = A.rows;
            int cols = A.cols;
            size_t shared_memory_size = (cols) * sizeof(Real);
            int persist_flag = persist_state ? 1 : 0;
            switch(threads_per_block){
                case 32:
                    chrKernel<Real, PRNGenerator, 32><<<blocks_per_grid, threads_per_block, shared_memory_size, stream>>>(A.dmat, slack.dmat, x0.dmat, samples.dmat, rows, cols, global_states, samples_per_chain, thinning, blocks_per_grid, persist_flag);
                    break;
                case 64:    
                    chrKernel<Real, PRNGenerator, 64><<<blocks_per_grid, threads_per_block, shared_memory_size, stream>>>(A.dmat, slack.dmat, x0.dmat, samples.dmat, rows, cols, global_states, samples_per_chain, thinning, blocks_per_grid, persist_flag);
                    break;
                case 128:
                    chrKernel<Real, PRNGenerator, 128><<<blocks_per_grid, threads_per_block, shared_memory_size, stream>>>(A.dmat, slack.dmat, x0.dmat, samples.dmat, rows, cols, global_states, samples_per_chain, thinning, blocks_per_grid, persist_flag);
                    break;
                case 256:
                    chrKernel<Real, PRNGenerator, 256><<<blocks_per_grid, threads_per_block, shared_memory_size, stream>>>(A.dmat, slack.dmat, x0.dmat, samples.dmat, rows, cols, global_states, samples_per_chain, thinning, blocks_per_grid, persist_flag);
                    break;
                case 512:
                    chrKernel<Real, PRNGenerator, 512><<<blocks_per_grid, threads_per_block, shared_memory_size, stream>>>(A.dmat, slack.dmat, x0.dmat, samples.dmat, rows, cols, global_states, samples_per_chain, thinning, blocks_per_grid, persist_flag);
                    break;
                case 1024:
                    chrKernel<Real, PRNGenerator, 1024><<<blocks_per_grid, threads_per_block, shared_memory_size, stream>>>(A.dmat, slack.dmat, x0.dmat, samples.dmat, rows, cols, global_states, samples_per_chain, thinning, blocks_per_grid, persist_flag);
                    break;
                default:
                    throw std::runtime_error("Invalid threads per block");
                    break;
                }
            CUDA_CHECK(cudaGetLastError());
        }

        template <typename PRNGenerator>
        __global__ void initPrngStatesCHR(PRNGenerator *states, int seed, int nstates){
            int tid = threadIdx.x + blockIdx.x * blockDim.x;
            if(tid < nstates)
                curand_init(seed,tid,0,&states[tid]);
        }

    }
}