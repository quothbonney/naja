#pragma once

#include <string>
#include <vector>

#include "runtime_config.h"

namespace naja::pipeline {

void write_generated_config(const std::string& path, const RuntimeConfig& cfg);

std::vector<std::string> load_model_list(const std::string& path);

}
