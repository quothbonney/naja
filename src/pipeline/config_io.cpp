#include "pipeline/config_io.h"

#include <fstream>
#include <iomanip>
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

} // namespace

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

}
