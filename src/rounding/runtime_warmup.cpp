#include "rounding/runtime_warmup.h"

#include <algorithm>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "rounding/jacobi.h"
#include "gpusamplers.h"

namespace naja::rounding {

PairSchedule estimate_runtime_pair_schedule(naja::gpu::DMatrix<double>& A_d,
                                            naja::gpu::DVector<double>& b_d,
                                            const Eigen::VectorXd& x0_host,
                                            int reduced_dim,
                                            int tpb_ss,
                                            int seed,
                                            int iter_rounding_passes,
                                            int iter_rounding_warmup,
                                            double pair_prob) {
    if (reduced_dim < 2) throw std::invalid_argument("estimate_runtime_pair_schedule: reduced_dim must be >= 2");
    if (iter_rounding_passes <= 0) throw std::invalid_argument("estimate_runtime_pair_schedule: passes must be > 0");
    if (iter_rounding_warmup <= 0) throw std::invalid_argument("estimate_runtime_pair_schedule: warmup must be > 0");

    std::vector<int> perm(reduced_dim);
    for (int i = 0; i < reduced_dim; ++i) perm[i] = i;
    std::mt19937_64 rng(static_cast<uint64_t>(seed) ^ 0x9e3779b97f4a7c15ULL);
    std::shuffle(perm.begin(), perm.end(), rng);

    const int n_pairs = reduced_dim / 2;
    PairSchedule sched;
    sched.i.resize(n_pairs);
    sched.j.resize(n_pairs);
    sched.c.resize(n_pairs);
    sched.s.resize(n_pairs);
    for (int k = 0; k < n_pairs; ++k) {
        sched.i[k] = perm[2 * k + 0];
        sched.j[k] = perm[2 * k + 1];
    }
    sched.c.setOnes();
    sched.s.setZero();

    const int warmup_chains = 1;
    Eigen::MatrixXd X0_warmup(reduced_dim, warmup_chains);
    X0_warmup.col(0) = x0_host;
    naja::gpu::DMatrix<double> X_warmup_d(X0_warmup);

    for (int pass = 0; pass < iter_rounding_passes; ++pass) {
        const int warmup_pair_mode = (pass == 0) ? 1 : 2;
        const double warmup_pair_prob = std::min(std::max(pair_prob, 0.05), 0.5);

        std::unique_ptr<naja::gpu::DVector<int>> pi_tmp;
        std::unique_ptr<naja::gpu::DVector<int>> pj_tmp;
        std::unique_ptr<naja::gpu::DVector<double>> pc_tmp;
        std::unique_ptr<naja::gpu::DVector<double>> ps_tmp;
        if (warmup_pair_mode == 2) {
            pi_tmp = std::make_unique<naja::gpu::DVector<int>>(sched.i);
            pj_tmp = std::make_unique<naja::gpu::DVector<int>>(sched.j);
            pc_tmp = std::make_unique<naja::gpu::DVector<double>>(sched.c);
            ps_tmp = std::make_unique<naja::gpu::DVector<double>>(sched.s);
        }

        auto warmup_samples_d = naja::gpu::CoordinateHitAndRun(
            A_d,
            b_d,
            X_warmup_d,
            iter_rounding_warmup,
            1,
            warmup_chains,
            tpb_ss,
            seed ^ (pass + 1),
            warmup_pair_prob,
            0,
            0.0,
            8,
            warmup_pair_mode,
            pi_tmp ? pi_tmp.get() : nullptr,
            pj_tmp ? pj_tmp.get() : nullptr,
            pc_tmp ? pc_tmp.get() : nullptr,
            ps_tmp ? ps_tmp.get() : nullptr
        );
        cudaDeviceSynchronize();
        Eigen::MatrixXd W = warmup_samples_d.toHost();

        for (int k = 0; k < n_pairs; ++k) {
            const int ii = sched.i[k];
            const int jj = sched.j[k];
            const auto vi = W.row(ii).array();
            const auto vj = W.row(jj).array();
            const double mi = vi.mean();
            const double mj = vj.mean();
            const auto di = vi - mi;
            const auto dj = vj - mj;
            const double var_i = (di * di).mean();
            const double var_j = (dj * dj).mean();
            const double cov_ij = (di * dj).mean();
            const auto cs = naja::rounding::jacobi_rotation_cs(var_i, var_j, cov_ij);
            sched.c[k] = cs.first;
            sched.s[k] = cs.second;
        }
    }

    return sched;
}

} // namespace naja::rounding

