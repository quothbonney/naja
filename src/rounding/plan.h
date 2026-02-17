#pragma once

#include <Eigen/Dense>

#include <memory>

#include "dmatrix.h"
#include "dvector.h"
#include "rounding/config.h"

namespace naja::rounding {

struct RoundingPlan {
    int pair_mode = 1;
    std::unique_ptr<naja::gpu::DVector<int>> pair_i_d;
    std::unique_ptr<naja::gpu::DVector<int>> pair_j_d;
    std::unique_ptr<naja::gpu::DVector<double>> pair_c_d;
    std::unique_ptr<naja::gpu::DVector<double>> pair_s_d;
};

RoundingPlan build_rounding_plan(const RoundingConfig& cfg,
                                 naja::gpu::DMatrix<double>& A_d,
                                 naja::gpu::DVector<double>& b_d,
                                 const Eigen::VectorXd& x0_host,
                                 int reduced_dim,
                                 int tpb_ss,
                                 int seed,
                                 bool status_enabled,
                                 bool verbose);

} // namespace naja::rounding


