#include "rounding/plan.h"

#include <algorithm>
#include <iostream>

#include "rounding/runtime_warmup.h"
#include "rounding/schedule_io.h"
#include "util/status.h"

namespace naja::rounding {

RoundingPlan build_rounding_plan(const RoundingConfig& cfg,
                                 naja::gpu::DMatrix<double>& A_d,
                                 naja::gpu::DVector<double>& b_d,
                                 const Eigen::VectorXd& x0_host,
                                 int reduced_dim,
                                 int tpb_ss,
                                 int seed,
                                 bool status_enabled,
                                 bool verbose) {
    validate_rounding_config(cfg);

    RoundingPlan plan;

    if (!cfg.pair_schedule_path.empty()) {
        naja::status::phase(status_enabled, "loading pair schedule");
        auto sched = load_pair_schedule_csv(cfg.pair_schedule_path);
        plan.pair_i_d = std::make_unique<naja::gpu::DVector<int>>(sched.i);
        plan.pair_j_d = std::make_unique<naja::gpu::DVector<int>>(sched.j);
        plan.pair_c_d = std::make_unique<naja::gpu::DVector<double>>(sched.c);
        plan.pair_s_d = std::make_unique<naja::gpu::DVector<double>>(sched.s);
        plan.pair_mode = 2;
        return plan;
    }

    if (cfg.iter_rounding_passes > 0 && cfg.iter_rounding_warmup > 0 && reduced_dim >= 2) {
        naja::status::phase(status_enabled, "iterative rounding-lite (warmup)");
        auto sched = estimate_runtime_pair_schedule(
            A_d,
            b_d,
            x0_host,
            reduced_dim,
            tpb_ss,
            seed,
            cfg.iter_rounding_passes,
            cfg.iter_rounding_warmup,
            cfg.pair_prob
        );
        plan.pair_i_d = std::make_unique<naja::gpu::DVector<int>>(sched.i);
        plan.pair_j_d = std::make_unique<naja::gpu::DVector<int>>(sched.j);
        plan.pair_c_d = std::make_unique<naja::gpu::DVector<double>>(sched.c);
        plan.pair_s_d = std::make_unique<naja::gpu::DVector<double>>(sched.s);
        plan.pair_mode = 2;
        if (verbose) {
            std::cout << "  iter_rounding_passes " << cfg.iter_rounding_passes << "\n";
            std::cout << "  iter_rounding_warmup " << cfg.iter_rounding_warmup << "\n";
            std::cout << "  iter_rounding_pairs  " << (reduced_dim / 2) << "\n";
        }
    }

    return plan;
}

} // namespace naja::rounding

