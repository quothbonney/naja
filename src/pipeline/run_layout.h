#pragma once

#include <string>

namespace naja::pipeline {

// Allocates and creates a unique run dir:
//   <out_root>/<run_name>_<YYYYMMDD>_<IDX>
std::string allocate_run_dir(const std::string& out_root, const std::string& run_name);

}
