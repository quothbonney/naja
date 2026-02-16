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
                                        int tpb_ss,
                                        int seed,
                                        double pair_prob,
                                        int resync_interval,
                                        double ksparse_prob,
                                        int ksparse_k,
                                        int pair_mode,
                                        const DVector<int>* pair_i,
                                        const DVector<int>* pair_j,
                                        const DVector<double>* pair_c,
                                        const DVector<double>* pair_s)
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
            initPrngStatesCHR<<<blocksPerGrid, threadsPerBlock>>>(gen.states, seed, nchains);
            CUDA_CHECK(cudaGetLastError());

            int threads_ss = (tpb_ss > 0) ? tpb_ss : 128;
            int blocks_ss = nchains;

            // Launch CHR kernel
            launchChrKernel(A_d, b_d, slack_d, X_d, samples_d, gen.states,
                            nspc, thinning, threads_ss, blocks_ss,
                            false,
                            static_cast<double>(pair_prob),
                            resync_interval,
                            static_cast<double>(ksparse_prob),
                            ksparse_k,
                            pair_mode,
                            pair_i,
                            pair_j,
                            pair_c,
                            pair_s,
                            0);


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
                                                       int tpb_ss,
                                                       int seed,
                                                       double pair_prob,
                                                       int resync_interval,
                                                       double ksparse_prob,
                                                       int ksparse_k,
                                                       int pair_mode,
                                                       const DVector<int>* pair_i,
                                                       const DVector<int>* pair_j,
                                                       const DVector<double>* pair_c,
                                                       const DVector<double>* pair_s)
        {
            DMatrix<double> samples_d = CoordinateHitAndRun(A_d, b_d, X_d, nspc, thinning, nchains, tpb_ss, seed, pair_prob, resync_interval, ksparse_prob, ksparse_k, pair_mode, pair_i, pair_j, pair_c, pair_s);

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
                                                int seed,
                                                double pair_prob,
                                                int resync_interval,
                                                double ksparse_prob,
                                                int ksparse_k,
                                                int pair_mode,
                                                const DVector<int>* pair_i,
                                                const DVector<int>* pair_j,
                                                const DVector<double>* pair_c,
                                                const DVector<double>* pair_s,
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
            initPrngStatesCHR<<<blocksPerGrid, threadsPerBlock, 0, compute_stream>>>(gen.states, seed, nchains);
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

                launchChrKernel(A_d, b_d, slack_d, X_d, dev_chunk[cur], gen.states,
                                nspc_this, thinning, tpb, blocks_ss,
                                true,
                                static_cast<double>(pair_prob),
                                resync_interval,
                                static_cast<double>(ksparse_prob),
                                ksparse_k,
                                pair_mode,
                                pair_i,
                                pair_j,
                                pair_c,
                                pair_s,
                                compute_stream);
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
        __global__ void chrKernel(Real* A,
                                  const Real* b,
                                  Real* slack,
                                  Real* X0,
                                  Real* samples,
                                  int rows,
                                  int cols,
                                  PRNGenerator* global_states,
                                  int samples_per_chain,
                                  int thinning,
                                  int nchains,
                                  int persist_state,
                                  Real pair_prob,
                                  int resync_interval,
                                  Real ksparse_prob,
                                  int ksparse_k,
                                  int pair_mode,
                                  const int* pair_i,
                                  const int* pair_j,
                                  const Real* pair_c,
                                  const Real* pair_s,
                                  int n_pairs){

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
            __shared__ int e_i_shared;
            __shared__ int e_j_shared;
            __shared__ int use_pair_shared;
            __shared__ Real ci_shared;
            __shared__ Real cj_shared;
            __shared__ int use_ksparse_shared;
            __shared__ int k_shared;
            __shared__ int ks_idx_shared[32];
            __shared__ Real ks_coef_shared[32];

            for (int t = 0; t < samples_per_chain * thinning; ++t){
                
                bool save = t % thinning == 0;
                int colOffset = static_cast<int>(t / thinning) * nchains;
                if (tid == 0) {
                    PRNGenerator localState = global_states[bid];
                    Real ucoord = curand_uniform_double(&localState);
                    const int want_ksparse = (ksparse_prob > Real(0.0) && ucoord < ksparse_prob) ? 1 : 0;
                    const int want_pair = (!want_ksparse && pair_prob > Real(0.0) && ucoord < (ksparse_prob + pair_prob)) ? 1 : 0;
                    int use_pair = want_pair;
                    int use_ksparse = want_ksparse;

                    // Default: coordinate step along e_i
                    int ei = static_cast<int>(curand_uniform_double(&localState) * static_cast<Real>(cols));
                    if (ei >= cols) ei = cols - 1;
                    int ej = -1;
                    Real ci = Real(1.0);
                    Real cj = Real(0.0);

                    // k-sparse direction: u has k nonzeros with +/- 1/sqrt(k).
                    // We cap k at 32 to keep shared memory simple (good enough as an "escape" move).
                    int k = ksparse_k;
                    if (k < 1) k = 1;
                    if (k > 32) k = 32;
                    if (k > cols) k = cols;
                    if (use_ksparse) {
                        const Real inv_sqrt_k = Real(1.0) / sqrt((Real)k);
                        // Sample k indices with replacement (cheap). Duplicates just reduce effective sparsity; OK for now.
                        for (int t = 0; t < k; ++t) {
                            int idx = static_cast<int>(curand_uniform_double(&localState) * static_cast<Real>(cols));
                            if (idx >= cols) idx = cols - 1;
                            const Real sign = (curand_uniform_double(&localState) < Real(0.5)) ? Real(1.0) : Real(-1.0);
                            ks_idx_shared[t] = idx;
                            ks_coef_shared[t] = sign * inv_sqrt_k;
                        }
                    } else {
                        // zero out first slot for safety
                        ks_idx_shared[0] = 0;
                        ks_coef_shared[0] = Real(0.0);
                    }

                    if (use_pair) {
                        if (cols < 2) {
                            use_pair = 0;
                        } else if (pair_mode == 1) {
                            // Fixed pair direction: u = (e_i - e_j)/sqrt(2)
                            int r = static_cast<int>(curand_uniform_double(&localState) * static_cast<Real>(cols - 1));
                            if (r >= cols - 1) r = cols - 2;
                            ej = (r >= ei) ? (r + 1) : r;
                            const Real inv_sqrt2 = Real(0.70710678118654752440);
                            ci = inv_sqrt2;
                            cj = -inv_sqrt2;
                        } else if (pair_mode == 2) {
                            // Rotated 2-sparse direction from precomputed Jacobi pairs.
                            if (n_pairs <= 0 || pair_i == nullptr || pair_j == nullptr || pair_c == nullptr || pair_s == nullptr) {
                                use_pair = 0;
                            } else {
                                int k = static_cast<int>(curand_uniform_double(&localState) * static_cast<Real>(n_pairs));
                                if (k >= n_pairs) k = n_pairs - 1;
                                ei = pair_i[k];
                                ej = pair_j[k];
                                Real c = pair_c[k];
                                Real s = pair_s[k];
                                // Choose one of the two orthonormal rotated axes in this plane:
                                // u1 = ( c) e_i + ( s) e_j
                                // u2 = (-s) e_i + ( c) e_j
                                const int which = (curand_uniform_double(&localState) < Real(0.5)) ? 0 : 1;
                                if (which == 0) {
                                    ci = c;
                                    cj = s;
                                } else {
                                    ci = -s;
                                    cj = c;
                                }
                                // Symmetrize direction distribution: sample +/- u with equal probability.
                                const Real sign = (curand_uniform_double(&localState) < Real(0.5)) ? Real(1.0) : Real(-1.0);
                                ci *= sign;
                                cj *= sign;
                            }
                        } else {
                            // Unknown mode -> fall back to coordinate.
                            use_pair = 0;
                        }
                    }

                    use_pair_shared = use_pair;
                    use_ksparse_shared = use_ksparse;
                    k_shared = k;
                    e_i_shared = ei;
                    e_j_shared = ej;
                    ci_shared = ci;
                    cj_shared = cj;
                    global_states[bid] = localState;
                }
                __syncthreads();
                int ei = e_i_shared;
                int ej = e_j_shared;
                int use_pair = use_pair_shared;
                Real ci = ci_shared;
                Real cj = cj_shared;
                int use_ksparse = use_ksparse_shared;
                int k = k_shared;

                // Optional numerical hygiene: resync slack = b - A*x every resync_interval iterations.
                // This is expensive; keep resync_interval == 0 unless you know you need it.
                if (resync_interval > 0 && (t % resync_interval) == 0) {
                    for (int r = tid; r < rows; r += threadstride) {
                        Real acc = Real(0.0);
                        for (int c = 0; c < cols; ++c) {
                            acc += A[IDX2C(r, c, rows)] * x_s[c];
                        }
                        slack[IDX2C(r, bid, rows)] = b[r] - acc;
                    }
                    __syncthreads();
                }

                Real partial_max = cuda::std::numeric_limits<Real>::lowest();
                Real partial_min = cuda::std::numeric_limits<Real>::max();
                for (int i = tid; i < rows; i += threadstride){

                    // Get slack and projected direction
                    Real s = slack[IDX2C(i, bid, rows)];
                    Real ae;
                    if (use_ksparse) {
                        Real acc = Real(0.0);
                        for (int t = 0; t < k; ++t) {
                            const int idx = ks_idx_shared[t];
                            const Real coef = ks_coef_shared[t];
                            acc += A[IDX2C(i, idx, rows)] * coef;
                        }
                        ae = acc;
                    } else if (!use_pair) {
                        ae = A[IDX2C(i, ei, rows)];
                    } else {
                        const Real a_i = A[IDX2C(i, ei, rows)];
                        const Real a_j = A[IDX2C(i, ej, rows)];
                        ae = a_i * ci + a_j * cj;
                    }
                    
                    // Legacy stable formulation for step size bounds.
                    Real inv_dist;
                    if (s == Real(0.0)) {
                        if (ae > Real(0.0)) inv_dist = cuda::std::numeric_limits<Real>::infinity();
                        else if (ae < Real(0.0)) inv_dist = -cuda::std::numeric_limits<Real>::infinity();
                        else inv_dist = Real(0.0);
                    } else {
                        inv_dist = ae / s;
                        inv_dist = isnan(inv_dist) ? Real(0.0) : inv_dist;
                    }
                    partial_max = fmax(partial_max, inv_dist);
                    partial_min = fmin(partial_min, inv_dist);
                }
                Real aggregate_max = BlockReduce(temp_storage_max).Reduce(partial_max, cub::Max());
                Real aggregate_min = BlockReduce(temp_storage_min).Reduce(partial_min, cub::Min());
                if(tid==0){
                    PRNGenerator localState = global_states[bid];
                    Real usample = curand_uniform_double(&localState);
                    alpha = (1/aggregate_min) + usample * ((1/aggregate_max) - (1/aggregate_min));
                    if (use_ksparse) {
                        // Update all k coordinates touched by the sparse direction.
                        for (int t = 0; t < k; ++t) {
                            const int idx = ks_idx_shared[t];
                            const Real coef = ks_coef_shared[t];
                            x_s[idx] += alpha * coef;
                        }
                    } else if (!use_pair) {
                        x_s[ei] += alpha;
                    } else {
                        x_s[ei] += alpha * ci;
                        x_s[ej] += alpha * cj;
                    }
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
                    Real ae;
                    if (use_ksparse) {
                        Real acc = Real(0.0);
                        for (int t = 0; t < k; ++t) {
                            const int idx = ks_idx_shared[t];
                            const Real coef = ks_coef_shared[t];
                            acc += A[IDX2C(i, idx, rows)] * coef;
                        }
                        ae = acc;
                    } else if (!use_pair) {
                        ae = A[IDX2C(i, ei, rows)];
                    } else {
                        const Real a_i = A[IDX2C(i, ei, rows)];
                        const Real a_j = A[IDX2C(i, ej, rows)];
                        ae = a_i * ci + a_j * cj;
                    }
                    slack[IDX2C(i, bid, rows)] -= alpha * ae;
                }
            }

            if (persist_state) {
                for (int i = tid; i < cols; i += threadstride) {
                    X0[IDX2C(i, bid, cols)] = x_s[i];
                }
            }
        }
        template __global__ void chrKernel<double, curandState, 32>(double*, const double*, double*, double*, double*, int, int, curandState*, int, int, int, int, double, int, double, int, int, const int*, const int*, const double*, const double*, int);
        template __global__ void chrKernel<double, curandState, 64>(double*, const double*, double*, double*, double*, int, int, curandState*, int, int, int, int, double, int, double, int, int, const int*, const int*, const double*, const double*, int);
        template __global__ void chrKernel<double, curandState, 128>(double*, const double*, double*, double*, double*, int, int, curandState*, int, int, int, int, double, int, double, int, int, const int*, const int*, const double*, const double*, int);
        template __global__ void chrKernel<double, curandState, 256>(double*, const double*, double*, double*, double*, int, int, curandState*, int, int, int, int, double, int, double, int, int, const int*, const int*, const double*, const double*, int);
        template __global__ void chrKernel<double, curandState, 512>(double*, const double*, double*, double*, double*, int, int, curandState*, int, int, int, int, double, int, double, int, int, const int*, const int*, const double*, const double*, int);
        template __global__ void chrKernel<double, curandState, 1024>(double*, const double*, double*, double*, double*, int, int, curandState*, int, int, int, int, double, int, double, int, int, const int*, const int*, const double*, const double*, int);

        template <typename Real, typename PRNGenerator>
        void launchChrKernel(DMatrix<Real>& A,
                             const DVector<Real>& b,
                             DMatrix<Real>& slack,
                             DMatrix<Real>& x0,
                             DMatrix<Real>& samples,
                             PRNGenerator* global_states,
                             int samples_per_chain,
                             int thinning,
                             int threads_per_block,
                             int blocks_per_grid,
                             bool persist_state,
                             Real pair_prob,
                             int resync_interval,
                             Real ksparse_prob,
                             int ksparse_k,
                             int pair_mode,
                             const DVector<int>* pair_i,
                             const DVector<int>* pair_j,
                             const DVector<Real>* pair_c,
                             const DVector<Real>* pair_s,
                             cudaStream_t stream){
            int rows = A.rows;
            int cols = A.cols;
            size_t shared_memory_size = (cols) * sizeof(Real);
            int persist_flag = persist_state ? 1 : 0;
            const int* pair_i_ptr = (pair_i ? pair_i->dvec : nullptr);
            const int* pair_j_ptr = (pair_j ? pair_j->dvec : nullptr);
            const Real* pair_c_ptr = (pair_c ? pair_c->dvec : nullptr);
            const Real* pair_s_ptr = (pair_s ? pair_s->dvec : nullptr);
            const int n_pairs = (pair_i ? pair_i->len : 0);
            switch(threads_per_block){
                case 32:
                    chrKernel<Real, PRNGenerator, 32><<<blocks_per_grid, threads_per_block, shared_memory_size, stream>>>(A.dmat, b.dvec, slack.dmat, x0.dmat, samples.dmat, rows, cols, global_states, samples_per_chain, thinning, blocks_per_grid, persist_flag, pair_prob, resync_interval, ksparse_prob, ksparse_k, pair_mode, pair_i_ptr, pair_j_ptr, pair_c_ptr, pair_s_ptr, n_pairs);
                    break;
                case 64:    
                    chrKernel<Real, PRNGenerator, 64><<<blocks_per_grid, threads_per_block, shared_memory_size, stream>>>(A.dmat, b.dvec, slack.dmat, x0.dmat, samples.dmat, rows, cols, global_states, samples_per_chain, thinning, blocks_per_grid, persist_flag, pair_prob, resync_interval, ksparse_prob, ksparse_k, pair_mode, pair_i_ptr, pair_j_ptr, pair_c_ptr, pair_s_ptr, n_pairs);
                    break;
                case 128:
                    chrKernel<Real, PRNGenerator, 128><<<blocks_per_grid, threads_per_block, shared_memory_size, stream>>>(A.dmat, b.dvec, slack.dmat, x0.dmat, samples.dmat, rows, cols, global_states, samples_per_chain, thinning, blocks_per_grid, persist_flag, pair_prob, resync_interval, ksparse_prob, ksparse_k, pair_mode, pair_i_ptr, pair_j_ptr, pair_c_ptr, pair_s_ptr, n_pairs);
                    break;
                case 256:
                    chrKernel<Real, PRNGenerator, 256><<<blocks_per_grid, threads_per_block, shared_memory_size, stream>>>(A.dmat, b.dvec, slack.dmat, x0.dmat, samples.dmat, rows, cols, global_states, samples_per_chain, thinning, blocks_per_grid, persist_flag, pair_prob, resync_interval, ksparse_prob, ksparse_k, pair_mode, pair_i_ptr, pair_j_ptr, pair_c_ptr, pair_s_ptr, n_pairs);
                    break;
                case 512:
                    chrKernel<Real, PRNGenerator, 512><<<blocks_per_grid, threads_per_block, shared_memory_size, stream>>>(A.dmat, b.dvec, slack.dmat, x0.dmat, samples.dmat, rows, cols, global_states, samples_per_chain, thinning, blocks_per_grid, persist_flag, pair_prob, resync_interval, ksparse_prob, ksparse_k, pair_mode, pair_i_ptr, pair_j_ptr, pair_c_ptr, pair_s_ptr, n_pairs);
                    break;
                case 1024:
                    chrKernel<Real, PRNGenerator, 1024><<<blocks_per_grid, threads_per_block, shared_memory_size, stream>>>(A.dmat, b.dvec, slack.dmat, x0.dmat, samples.dmat, rows, cols, global_states, samples_per_chain, thinning, blocks_per_grid, persist_flag, pair_prob, resync_interval, ksparse_prob, ksparse_k, pair_mode, pair_i_ptr, pair_j_ptr, pair_c_ptr, pair_s_ptr, n_pairs);
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