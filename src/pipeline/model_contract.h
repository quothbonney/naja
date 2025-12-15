#pragma once

#include <string>

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

}
