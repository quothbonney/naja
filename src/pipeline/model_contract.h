#pragma once

#include <string>
#include <vector>

#include "runtime_config.h"

namespace naja::pipeline {

struct ModelContract {
    std::string model_dir;
    std::string rounding_dir;
    std::string gem_dir;
    std::string model_name;
};

ModelContract parse_model_dir(const std::string& model_dir);

// Validates the on-disk rounding contract. Throws on any error.
void validate_contract(const ModelContract& c, bool backmap);

struct CsvShape {
    int rows = 0;
    int cols = 0;
};

CsvShape csv_shape(const std::string& path);
int text_line_count(const std::string& path);

void write_generated_config(const std::string& path, const RuntimeConfig& cfg);
std::vector<std::string> load_model_list(const std::string& path);

std::string allocate_run_dir(const std::string& out_root, const std::string& run_name);

}
