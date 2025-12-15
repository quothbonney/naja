#pragma once

#include <string>
#include <vector>

#include "pipeline/model_contract.h"
#include "runtime_config.h"

namespace naja::pipeline {

// Writes run_manifest.json into cfg.OUT_DIR.
// `full_argv` should be the exact command tokens starting at "naja", e.g.:
//   ["naja","sample","run",...]
void write_run_manifest(const RuntimeConfig& cfg,
                        const ModelContract* model,
                        const std::vector<std::string>& full_argv);

} // namespace naja::pipeline


