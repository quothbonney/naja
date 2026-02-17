#pragma once

#include <string>

#include "pipeline/model_contract.h"

namespace naja::rounding {

void normalize_extra_constraints(const naja::pipeline::ModelContract& c);

void inherit_rounding_impl(const naja::pipeline::ModelContract& base,
                           const naja::pipeline::ModelContract& target,
                           const std::string& mode);

} // namespace naja::rounding


