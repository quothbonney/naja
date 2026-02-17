#include "pipeline/model_contract.h"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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

int count_cols_csv_line(const std::string& line) {
    if (line.empty()) return 0;
    int cols = 1;
    for (char c : line) {
        if (c == ',') cols++;
    }
    return cols;
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

CsvShape csv_shape(const std::string& path) {
    if (!path_exists(path)) {
        throw std::runtime_error("missing file: " + path);
    }
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("cannot open file: " + path);
    }
    CsvShape s;
    std::string line;
    bool first = true;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (first) {
            s.cols = count_cols_csv_line(line);
            first = false;
        }
        s.rows++;
    }
    if (s.rows == 0 || s.cols == 0) {
        throw std::runtime_error("empty csv: " + path);
    }
    return s;
}

int text_line_count(const std::string& path) {
    if (!path_exists(path)) {
        throw std::runtime_error("missing file: " + path);
    }
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("cannot open file: " + path);
    }
    int n = 0;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty()) n++;
    }
    if (n == 0) {
        throw std::runtime_error("empty file: " + path);
    }
    return n;
}

void write_generated_config(const std::string& path, const RuntimeConfig& cfg) {
    std::ofstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("cannot write generated config: " + path);
    }
    f << "DATA_DIR=" << cfg.DATA_DIR << "\n";
    f << "MODEL_NAME=" << cfg.MODEL_NAME << "\n";
    f << "OUT_DIR=" << cfg.OUT_DIR << "\n";
    f << "N_CHAINS=" << cfg.N_CHAINS << "\n";
    f << "N_SAMPLES=" << cfg.N_SAMPLES << "\n";
    f << "TPB_SS=" << cfg.TPB_SS << "\n";
    f << "GPU_DEVICE=" << cfg.GPU_DEVICE << "\n";
    f << "BACK_TRANSFORM=" << (cfg.BACK_TRANSFORM ? "true" : "false") << "\n";
    f << "WRITE_DATA=" << (cfg.WRITE_DATA ? "true" : "false") << "\n";
    f << "VERBOSE=" << (cfg.VERBOSE ? "true" : "false") << "\n";
    f << "STATUS=" << (cfg.STATUS ? "true" : "false") << "\n";
    if (cfg.THINNING > 0) {
        f << "THINNING=" << cfg.THINNING << "\n";
    }
    if (cfg.PAIR_PROB > 0.0) {
        f << "PAIR_PROB=" << std::setprecision(12) << cfg.PAIR_PROB << "\n";
    }
    if (cfg.RESYNC_INTERVAL > 0) {
        f << "RESYNC_INTERVAL=" << cfg.RESYNC_INTERVAL << "\n";
    }
    if (cfg.ITER_ROUNDING_PASSES > 0) {
        f << "ITER_ROUNDING_PASSES=" << cfg.ITER_ROUNDING_PASSES << "\n";
    }
    if (cfg.ITER_ROUNDING_WARMUP > 0) {
        f << "ITER_ROUNDING_WARMUP=" << cfg.ITER_ROUNDING_WARMUP << "\n";
    }
    if (cfg.EXTRA_CONSTRAINT_EPS > 0.0) {
        f << "EXTRA_CONSTRAINT_EPS=" << std::setprecision(12) << cfg.EXTRA_CONSTRAINT_EPS << "\n";
    }
    if (cfg.CONSTRAINT_EPS > 0.0) {
        f << "CONSTRAINT_EPS=" << std::setprecision(12) << cfg.CONSTRAINT_EPS << "\n";
    }
    if (!cfg.PAIR_SCHEDULE.empty()) {
        f << "PAIR_SCHEDULE=" << cfg.PAIR_SCHEDULE << "\n";
    }
    if (!cfg.EXTRA_CONSTRAINTS.empty()) {
        f << "EXTRA_CONSTRAINTS=" << cfg.EXTRA_CONSTRAINTS << "\n";
    }
    if (!cfg.START_POLICY.empty()) {
        f << "START_POLICY=" << cfg.START_POLICY << "\n";
    }
    if (cfg.AFFINE_HULL_TOL > 0.0) {
        f << "AFFINE_HULL_TOL=" << std::setprecision(12) << cfg.AFFINE_HULL_TOL << "\n";
    }
    if (cfg.KSPARSE_PROB > 0.0) {
        f << "KSPARSE_PROB=" << std::setprecision(12) << cfg.KSPARSE_PROB << "\n";
    }
    if (cfg.KSPARSE_K != 8) {
        f << "KSPARSE_K=" << cfg.KSPARSE_K << "\n";
    }
    if (cfg.BOUNDS_FILTER) {
        f << "BOUNDS_POLICY=filter\n";
        f << "BOUNDS_EPS=" << std::setprecision(12) << cfg.BOUNDS_EPS << "\n";
        f << "WRITE_SAMPLES_VALID=" << (cfg.WRITE_SAMPLES_VALID ? "true" : "false") << "\n";
    } else {
        f << "BOUNDS_POLICY=ignore\n";
    }
    if (!cfg.GPU_LIST.empty()) f << "GPU_LIST=" << cfg.GPU_LIST << "\n";
    if (!cfg.BULK_MODEL_LIST.empty()) f << "BULK_MODEL_LIST=" << cfg.BULK_MODEL_LIST << "\n";
}

std::vector<std::string> load_model_list(const std::string& path) {
    require_nonempty_file(path, "model list");
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("cannot open model list: " + path);
    }
    std::vector<std::string> models;
    std::string line;
    while (std::getline(f, line)) {
        auto is_ws = [](unsigned char ch) { return ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n'; };
        while (!line.empty() && is_ws((unsigned char)line.front())) line.erase(line.begin());
        while (!line.empty() && is_ws((unsigned char)line.back())) line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        models.push_back(line);
    }
    if (models.empty()) {
        throw std::runtime_error("no models in list: " + path);
    }
    return models;
}

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
