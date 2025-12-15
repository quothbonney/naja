#pragma once

#include <Eigen/Dense>

#include "runtime_config.h"

namespace naja::engine {

// Writes:
// - valid_mask.npy (uint8 0/1)
// - valid_fraction.txt
// - bounds_report.json
// - samples_valid.npy (optional)
//
// Requires:
// - cfg.BACK_TRANSFORM == true
// - model_dir/gem/{reaction_ids.txt,l_bounds.csv,u_bounds.csv} exist and are non-empty
void bounds_filter_and_write(const RuntimeConfig& cfg, const Eigen::MatrixXd& samples_out);

}
