#include "rounding/inherit.h"

#include <fstream>
#include <stdexcept>
#include <string>

#include <unistd.h>

#include "cli/sample/common.h"
#include "utils.h"

namespace naja::rounding {

void normalize_extra_constraints(const naja::pipeline::ModelContract& c) {
    const std::string extra_A = c.rounding_dir + "/" + c.model_name + "_rounding_extra_A.csv";
    const std::string extra_b = c.rounding_dir + "/" + c.model_name + "_rounding_extra_b.csv";
    bool ha = path_exists(extra_A);
    bool hb = path_exists(extra_b);
    if (ha != hb) {
        throw std::runtime_error("extra constraints must be both-present or both-absent: " + extra_A + " / " + extra_b);
    }
    if (!ha) return;

    if (naja::cli::sample::file_is_empty(extra_A) || naja::cli::sample::file_is_empty(extra_b)) {
        naja::cli::sample::remove_if_exists(extra_A);
        naja::cli::sample::remove_if_exists(extra_b);
    }
}

void inherit_rounding_impl(const naja::pipeline::ModelContract& base,
                           const naja::pipeline::ModelContract& target,
                           const std::string& mode) {
    if (mode != "symlink" && mode != "copy") {
        throw std::runtime_error("invalid inherit mode: " + mode);
    }
    std::string base_round = base.model_dir + "/rounding";
    std::string target_round = target.model_dir + "/rounding";
    ensure_dir(target_round);

    const char* suffixes[] = {"A.csv", "b.csv", "T.csv", "shift.csv", "start.csv"};
    for (const char* suf : suffixes) {
        std::string src = base_round + "/" + base.model_name + "_rounding_" + suf;
        std::string dst = target_round + "/" + target.model_name + "_rounding_" + suf;
        naja::cli::sample::require_nonempty_file(src, std::string("base rounding ") + suf);
        naja::cli::sample::remove_if_exists(dst);

        if (mode == "symlink") {
            if (symlink(src.c_str(), dst.c_str()) != 0) {
                throw std::runtime_error("symlink failed: " + dst + " <- " + src);
            }
        } else {
            std::ifstream in(src, std::ios::binary);
            std::ofstream out(dst, std::ios::binary);
            if (!in.is_open() || !out.is_open()) {
                throw std::runtime_error("copy failed: " + dst + " <- " + src);
            }
            out << in.rdbuf();
        }
    }

    std::string prov = target_round + "/INHERITED_FROM.txt";
    std::ofstream f(prov);
    if (!f.is_open()) {
        throw std::runtime_error("cannot write provenance file: " + prov);
    }
    f << "base_model_dir=" << base.model_dir << "\n";
    f << "target_model_dir=" << target.model_dir << "\n";
    f << "mode=" << mode << "\n";
    f << "created_at=" << current_timestamp() << "\n";
}

} // namespace naja::rounding


