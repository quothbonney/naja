#include "cli/sample/commands.h"

#include <iostream>
#include <string>

#include "cli/sample/common.h"
#include "rounding/inherit.h"
#include "pipeline/model_contract.h"

namespace naja::cli::sample {

void cmd_inherit_rounding(int argc, char** argv) {
    std::string base_model_dir;
    std::string target_model_dir;
    std::string mode = "symlink";

    for (int i = 0; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--base-model-dir") base_model_dir = next_arg(i, argc, argv, a);
        else if (a == "--target-model-dir") target_model_dir = next_arg(i, argc, argv, a);
        else if (a == "--mode") mode = next_arg(i, argc, argv, a);
        else die_usage("unknown flag: " + a);
    }
    if (base_model_dir.empty()) die_usage("missing --base-model-dir");
    if (target_model_dir.empty()) die_usage("missing --target-model-dir");
    if (mode != "symlink" && mode != "copy") die_usage("invalid --mode: " + mode);

    naja::pipeline::ModelContract base = naja::pipeline::parse_model_dir(base_model_dir);
    naja::pipeline::ModelContract target = naja::pipeline::parse_model_dir(target_model_dir);

    naja::rounding::inherit_rounding_impl(base, target, mode);
    std::cout << "OK\n";
    std::cout << "base_model_dir   :: " << base.model_dir << "\n";
    std::cout << "target_model_dir :: " << target.model_dir << "\n";
}

} // namespace naja::cli::sample


