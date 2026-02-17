#pragma once

#include <string>
#include <vector>

#include "cli/sample/common.h"

namespace naja::condition {

struct ConditionCommonArgs {
    std::string base_model_dir;
    std::string out_model_dir;
    std::string row_id;
    std::string reaction_scores;
};

inline std::vector<std::string> build_condition_cmdline(const char* subcommand, int argc, char** argv) {
    std::vector<std::string> cmdline;
    cmdline.reserve((size_t)argc + 3);
    cmdline.push_back("naja");
    cmdline.push_back("condition");
    cmdline.push_back(subcommand);
    for (int i = 0; i < argc; ++i) cmdline.push_back(argv[i]);
    return cmdline;
}

inline bool parse_condition_common_flag(const std::string& arg,
                                        int& i,
                                        int argc,
                                        char** argv,
                                        ConditionCommonArgs& out) {
    if (arg == "--base-model-dir") {
        out.base_model_dir = naja::cli::sample::next_arg(i, argc, argv, arg);
        return true;
    }
    if (arg == "--out-model-dir") {
        out.out_model_dir = naja::cli::sample::next_arg(i, argc, argv, arg);
        return true;
    }
    if (arg == "--row-id") {
        out.row_id = naja::cli::sample::next_arg(i, argc, argv, arg);
        return true;
    }
    if (arg == "--reaction-scores") {
        out.reaction_scores = naja::cli::sample::next_arg(i, argc, argv, arg);
        return true;
    }
    return false;
}

} // namespace naja::condition


