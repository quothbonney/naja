#include "cli/sample/commands.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/stat.h>

#include "cli/sample/common.h"
#include "utils.h"

namespace naja::cli::sample {
namespace fs = std::filesystem;

namespace {

std::regex glob_to_regex(const std::string& glob) {
    std::string r;
    r.reserve(glob.size() * 2);
    r.push_back('^');
    for (char c : glob) {
        switch (c) {
            case '*': r += ".*"; break;
            case '?': r += "."; break;
            case '.': case '(': case ')': case '+': case '|': case '^': case '$': case '{': case '}': case '[': case ']': case '\\':
                r.push_back('\\'); r.push_back(c); break;
            default: r.push_back(c); break;
        }
    }
    r.push_back('$');
    return std::regex(r);
}

bool file_nonempty(const std::string& path) {
    if (!path_exists(path)) return false;
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return false;
    return st.st_size > 0;
}

bool has_gem(const std::string& model_dir) {
    const std::string gem = model_dir + "/gem";
    if (!is_directory(gem)) return false;
    if (!file_nonempty(gem + "/reaction_ids.txt")) return false;
    if (!file_nonempty(gem + "/l_bounds.csv")) return false;
    if (!file_nonempty(gem + "/u_bounds.csv")) return false;
    return true;
}

bool has_rounding(const std::string& model_dir, const std::string& model_name) {
    const std::string r = model_dir + "/rounding";
    if (!is_directory(r)) return false;
    if (!file_nonempty(r + "/" + model_name + "_rounding_A.csv")) return false;
    if (!file_nonempty(r + "/" + model_name + "_rounding_b.csv")) return false;
    if (!file_nonempty(r + "/" + model_name + "_rounding_start.csv")) return false;
    return true;
}

} // namespace

void cmd_list(int argc, char** argv) {
    std::string models_root;
    std::string prefix;
    std::string glob;
    bool filter_gem = false;
    bool filter_rounding = false;
    std::string out_path;

    for (int i = 0; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--models-root") models_root = next_arg(i, argc, argv, a);
        else if (a == "--prefix") prefix = next_arg(i, argc, argv, a);
        else if (a == "--glob") glob = next_arg(i, argc, argv, a);
        else if (a == "--has-gem") filter_gem = true;
        else if (a == "--has-rounding") filter_rounding = true;
        else if (a == "--out") out_path = next_arg(i, argc, argv, a);
        else die_usage("unknown flag: " + a);
    }

    if (models_root.empty()) die_usage("missing --models-root");
    if ((prefix.empty() && glob.empty()) || (!prefix.empty() && !glob.empty())) {
        die_usage("need exactly one of --prefix or --glob");
    }
    if (!is_directory(models_root)) {
        throw std::runtime_error("models-root is not a directory: " + models_root);
    }

    std::vector<std::string> names;
    std::regex rx;
    bool use_glob = !glob.empty();
    if (use_glob) rx = glob_to_regex(glob);

    for (const auto& ent : fs::directory_iterator(models_root)) {
        if (!ent.is_directory()) continue;
        std::string name = ent.path().filename().string();
        if (!prefix.empty()) {
            if (name.rfind(prefix, 0) != 0) continue;
        } else {
            if (!std::regex_match(name, rx)) continue;
        }

        std::string mdir = (fs::path(models_root) / name).string();
        if (filter_gem && !has_gem(mdir)) continue;
        if (filter_rounding && !has_rounding(mdir, name)) continue;
        names.push_back(name);
    }

    std::sort(names.begin(), names.end());

    if (!out_path.empty()) {
        std::ofstream out(out_path);
        if (!out.is_open()) throw std::runtime_error("cannot write: " + out_path);
        for (const auto& n : names) out << n << "\n";
        return;
    }

    for (const auto& n : names) {
        std::cout << n << "\n";
    }
}

} // namespace naja::cli::sample


