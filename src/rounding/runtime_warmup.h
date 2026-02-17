#pragma once

#include <Eigen/Dense>

#include "dmatrix.h"
#include "dvector.h"
#include "rounding/schedule_io.h"

namespace naja::rounding {

PairSchedule estimate_runtime_pair_schedule(naja::gpu::DMatrix<double>& A_d,
                                            naja::gpu::DVector<double>& b_d,
                                            const Eigen::VectorXd& x0_host,
                                            int reduced_dim,
                                            int tpb_ss,
                                            int seed,
                                            int iter_rounding_passes,
                                            int iter_rounding_warmup,
                                            double pair_prob);

} // namespace naja::rounding

