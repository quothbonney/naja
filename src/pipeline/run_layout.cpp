#include "pipeline/run_layout.h"

#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

#include "utils.h"

namespace naja::pipeline {

std::string allocate_run_dir(const std::string& out_root, const std::string& run_name) {
    ensure_dir(out_root);
    std::string date = current_datestamp_compact();
    for (int idx = 1; idx < 1000000; ++idx) {
        std::ostringstream oss;
        oss << out_root << "/" << run_name << "_" << date << "_" << std::setw(3) << std::setfill('0') << idx;
        std::string cand = oss.str();
        if (!path_exists(cand)) {
            ensure_dir(cand);
            return cand;
        }
    }
    throw std::runtime_error("could not allocate run directory under " + out_root);
}

}
