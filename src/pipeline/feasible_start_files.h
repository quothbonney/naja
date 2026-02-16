#pragma once

#include "pipeline/model_contract.h"

namespace naja::pipeline {

// If extra constraints are present for this model, compute a feasible start point for the
// augmented constraint system (A + extra_A) and overwrite `<model>_rounding_start.csv`.
// If extra constraints are absent, this is a no-op.
// Throws on any failure (missing files, LP failure, infeasible result).
void ensure_feasible_rounding_start_if_extra_present(const ModelContract& c);

} // namespace naja::pipeline


