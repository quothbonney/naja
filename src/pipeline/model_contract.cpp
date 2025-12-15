#include "pipeline/model_contract.h"

#include <stdexcept>
#include <string>

#include <sys/stat.h>

#include "utils.h"

namespace naja::pipeline {
namespace {

void require_nonempty_file(const std::string& path, const std::string& what) {
    if (!path_exists(path)) {
        throw std::runtime_error("missing " + what + ": " + path);
    }
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        throw std::runtime_error("cannot stat " + what + ": " + path);
    }
    if (st.st_size == 0) {
        throw std::runtime_error("empty " + what + ": " + path);
    }
}

void ensure_file_absent_or_nonempty_pair(const std::string& a, const std::string& b) {
    bool ha = path_exists(a);
    bool hb = path_exists(b);
    if (ha != hb) {
        throw std::runtime_error("extra constraints must be both-present or both-absent: " + a + " / " + b);
    }
    if (ha) {
        require_nonempty_file(a, "extra_A");
        require_nonempty_file(b, "extra_b");
    }
}

} // namespace

ModelContract parse_model_dir(const std::string& model_dir) {
    std::string md = model_dir;
    if (!md.empty() && md.back() == '/') md.pop_back();
    if (!is_directory(md)) {
        throw std::runtime_error("model-dir is not a directory: " + md);
    }
    auto slash = md.find_last_of('/');
    std::string name = (slash == std::string::npos) ? md : md.substr(slash + 1);
    ModelContract c;
    c.model_dir = md;
    c.rounding_dir = md + "/rounding";
    c.gem_dir = md + "/gem";
    c.model_name = name;
    return c;
}

void validate_contract(const ModelContract& c, bool backmap) {
    if (!is_directory(c.rounding_dir)) {
        throw std::runtime_error("missing rounding/ dir: " + c.rounding_dir);
    }
    require_nonempty_file(c.rounding_dir + "/" + c.model_name + "_rounding_A.csv", "rounding_A");
    require_nonempty_file(c.rounding_dir + "/" + c.model_name + "_rounding_b.csv", "rounding_b");
    require_nonempty_file(c.rounding_dir + "/" + c.model_name + "_rounding_start.csv", "rounding_start");
    if (backmap) {
        require_nonempty_file(c.rounding_dir + "/" + c.model_name + "_rounding_T.csv", "rounding_T");
        require_nonempty_file(c.rounding_dir + "/" + c.model_name + "_rounding_shift.csv", "rounding_shift");
    }
    ensure_file_absent_or_nonempty_pair(
        c.rounding_dir + "/" + c.model_name + "_rounding_extra_A.csv",
        c.rounding_dir + "/" + c.model_name + "_rounding_extra_b.csv"
    );
}

}
