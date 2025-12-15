#include "pipeline/run_manifest.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#include "utils.h"

namespace naja::pipeline {
namespace {

std::string read_all(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("cannot read file: " + path);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string exec_capture(const std::string& cmd) {
    std::array<char, 4096> buf{};
    std::string out;
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) {
        throw std::runtime_error("popen failed for: " + cmd);
    }
    while (true) {
        size_t n = fread(buf.data(), 1, buf.size(), p);
        if (n > 0) out.append(buf.data(), buf.data() + n);
        if (n < buf.size()) break;
    }
    int rc = pclose(p);
    (void)rc;
    while (!out.empty() && (out.back() == '\n' || out.back() == ' ' || out.back() == '\t')) out.pop_back();
    return out;
}

std::string binary_path() {
    std::array<char, 4096> buf{};
    ssize_t n = readlink("/proc/self/exe", buf.data(), (ssize_t)buf.size() - 1);
    if (n <= 0) {
        throw std::runtime_error("cannot resolve /proc/self/exe");
    }
    buf[(size_t)n] = 0;
    return std::string(buf.data());
}

std::string sha256sum(const std::string& path) {
    std::string cmd = "sha256sum " + path + " 2>/dev/null | awk '{print $1}'";
    std::string out = exec_capture(cmd);
    if (out.empty()) {
        throw std::runtime_error("cannot compute sha256sum for: " + path);
    }
    return out;
}

std::string git_head_or_empty(const std::string& cwd) {
    std::string cmd = "git -C " + cwd + " rev-parse HEAD 2>/dev/null";
    try {
        return exec_capture(cmd);
    } catch (...) {
        return {};
    }
}

void json_string(std::ostream& o, const std::string& s) {
    o << "\"";
    for (char c : s) {
        switch (c) {
            case '\\': o << "\\\\"; break;
            case '"': o << "\\\""; break;
            case '\n': o << "\\n"; break;
            case '\r': o << "\\r"; break;
            case '\t': o << "\\t"; break;
            default: o << c; break;
        }
    }
    o << "\"";
}

} // namespace

void write_run_manifest(const RuntimeConfig& cfg,
                        const ModelContract* model,
                        const std::vector<std::string>& full_argv) {
    const std::string out_path = cfg.OUT_DIR + "/run_manifest.json";

    const std::string exe = binary_path();
    struct stat st{};
    if (stat(exe.c_str(), &st) != 0) {
        throw std::runtime_error("cannot stat binary: " + exe);
    }

    // best-effort: repo root may not be a git checkout
    std::string git_head = git_head_or_empty(std::string("."));
    if (git_head.empty()) {
        auto slash = exe.find_last_of('/');
        if (slash != std::string::npos) {
            git_head = git_head_or_empty(exe.substr(0, slash));
        }
    }

    std::string inherited_from_path;
    std::string inherited_from;
    if (model) {
        const std::string p = model->rounding_dir + "/INHERITED_FROM.txt";
        if (path_exists(p)) {
            inherited_from_path = make_absolute_path(p);
            inherited_from = read_all(p);
        }
    }

    std::ofstream f(out_path);
    if (!f.is_open()) {
        throw std::runtime_error("cannot write run manifest: " + out_path);
    }

    f << "{\n";
    f << "  \"argv\": [";
    for (size_t i = 0; i < full_argv.size(); ++i) {
        if (i) f << ", ";
        json_string(f, full_argv[i]);
    }
    f << "],\n";

    f << "  \"model_dir\": ";
    if (model) json_string(f, make_absolute_path(model->model_dir)); else f << "null";
    f << ",\n";

    f << "  \"engine\": {\n";
    f << "    \"path\": ";
    json_string(f, make_absolute_path(exe));
    f << ",\n";
    f << "    \"sha256\": ";
    json_string(f, sha256sum(exe));
    f << ",\n";
    f << "    \"size\": " << (long long)st.st_size << "\n";
    f << "  },\n";

    f << "  \"git_head\": ";
    if (!git_head.empty()) json_string(f, git_head); else f << "null";
    f << ",\n";

    f << "  \"config\": {\n";
    f << "    \"gpu_device\": " << cfg.GPU_DEVICE << ",\n";
    f << "    \"n_chains\": " << cfg.N_CHAINS << ",\n";
    f << "    \"n_samples\": " << cfg.N_SAMPLES << ",\n";
    f << "    \"tpb_ss\": " << cfg.TPB_SS << ",\n";
    f << "    \"back_transform\": " << (cfg.BACK_TRANSFORM ? "true" : "false") << ",\n";
    f << "    \"write_data\": " << (cfg.WRITE_DATA ? "true" : "false") << ",\n";
    f << "    \"bounds_policy\": ";
    json_string(f, cfg.BOUNDS_FILTER ? "filter" : "ignore");
    f << ",\n";
    f << "    \"bounds_eps\": " << std::setprecision(12) << cfg.BOUNDS_EPS << ",\n";
    f << "    \"write_samples_valid\": " << (cfg.WRITE_SAMPLES_VALID ? "true" : "false") << "\n";
    f << "  },\n";

    f << "  \"resolved_paths\": {\n";
    f << "    \"model_dir\": ";
    json_string(f, make_absolute_path(cfg.MODEL_DIR));
    f << ",\n";
    f << "    \"A\": "; json_string(f, make_absolute_path(cfg.A_FILE)); f << ",\n";
    f << "    \"b\": "; json_string(f, make_absolute_path(cfg.B_FILE)); f << ",\n";
    f << "    \"start\": "; json_string(f, make_absolute_path(cfg.START_FILE)); f << ",\n";
    f << "    \"T\": "; json_string(f, make_absolute_path(cfg.T_FILE)); f << ",\n";
    f << "    \"shift\": "; json_string(f, make_absolute_path(cfg.SHIFT_FILE)); f << ",\n";
    f << "    \"extra_A\": "; json_string(f, make_absolute_path(cfg.A_EXTRA_FILE)); f << ",\n";
    f << "    \"extra_b\": "; json_string(f, make_absolute_path(cfg.B_EXTRA_FILE)); f << "\n";
    f << "  },\n";

    f << "  \"rounding_provenance\": {\n";
    f << "    \"path\": ";
    if (!inherited_from_path.empty()) json_string(f, inherited_from_path); else f << "null";
    f << ",\n";
    f << "    \"contents\": ";
    if (!inherited_from.empty()) json_string(f, inherited_from); else f << "null";
    f << "\n";
    f << "  }\n";

    f << "}\n";
}

}


