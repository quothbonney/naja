#include "cli/sample/commands.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli/sample/common.h"
#include "job.h"
#include "pipeline/config_io.h"
#include "pipeline/model_contract.h"
#include "pipeline/run_manifest.h"
#include "pipeline/run_layout.h"
#include "runtime_config.h"
#include "util/status.h"
#include "utils.h"

namespace naja::cli::sample {

void cmd_run(int argc, char** argv) {
    std::string model_dir;
    std::string out_root;
    int gpu = 0;
    int n_chains = -1;
    int n_samples = -1;
    int tpb = 128;
    bool backmap = false;
    bool write_npy = false;
    bool verbose = false;
    std::string bounds_policy = "ignore";
    double bounds_eps = 1e-6;
    bool write_samples_valid = false;
    bool quiet = false;
    bool dry_run = false;
    bool print_config = false;

    for (int i = 0; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model-dir") model_dir = next_arg(i, argc, argv, a);
        else if (a == "--out-root") out_root = next_arg(i, argc, argv, a);
        else if (a == "--gpu") gpu = std::stoi(next_arg(i, argc, argv, a));
        else if (a == "--n-chains") n_chains = std::stoi(next_arg(i, argc, argv, a));
        else if (a == "--n-samples") n_samples = std::stoi(next_arg(i, argc, argv, a));
        else if (a == "--tpb") tpb = std::stoi(next_arg(i, argc, argv, a));
        else if (a == "--backmap") backmap = true;
        else if (a == "--write-npy") write_npy = true;
        else if (a == "--verbose") verbose = true;
        else if (a == "--bounds-policy") bounds_policy = next_arg(i, argc, argv, a);
        else if (a == "--bounds-eps") bounds_eps = std::stod(next_arg(i, argc, argv, a));
        else if (a == "--write-samples-valid") write_samples_valid = true;
        else if (a == "--quiet") quiet = true;
        else if (a == "--dry-run") dry_run = true;
        else if (a == "--print-config") print_config = true;
        else die_usage("unknown flag: " + a);
    }
    if (model_dir.empty()) die_usage("missing --model-dir");
    if (out_root.empty()) die_usage("missing --out-root");
    if (n_chains <= 0) die_usage("invalid --n-chains");
    if (n_samples <= 0) die_usage("invalid --n-samples");
    if (bounds_policy != "ignore" && bounds_policy != "filter") die_usage("invalid --bounds-policy: " + bounds_policy);
    if (bounds_policy == "filter" && !backmap) {
        throw std::runtime_error("bounds-policy=filter requires --backmap");
    }

    RuntimeConfig cfg;
    cfg.TPB_SS = tpb;
    cfg.GPU_DEVICE = gpu;
    cfg.N_CHAINS = n_chains;
    cfg.N_SAMPLES = n_samples;
    cfg.BACK_TRANSFORM = backmap;
    cfg.WRITE_DATA = write_npy;
    cfg.VERBOSE = verbose;
    cfg.STATUS = !quiet;
    cfg.BOUNDS_FILTER = (bounds_policy == "filter");
    cfg.BOUNDS_EPS = bounds_eps;
    cfg.WRITE_SAMPLES_VALID = write_samples_valid;

    naja::pipeline::ModelContract c = naja::pipeline::parse_model_dir(model_dir);
    naja::pipeline::validate_contract(c, backmap);

    const std::string inherited = c.rounding_dir + "/INHERITED_FROM.txt";
    if (path_exists(inherited)) {
        std::cout << "rounding_provenance :: " << inherited << "\n";
        std::ifstream f(inherited);
        if (!f.is_open()) {
            throw std::runtime_error("cannot read rounding provenance: " + inherited);
        }
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty()) std::cout << "  " << line << "\n";
        }
    }

    auto slash = c.model_dir.find_last_of('/');
    cfg.DATA_DIR = (slash == std::string::npos) ? "." : c.model_dir.substr(0, slash);
    cfg.MODEL_NAME = c.model_name;
    cfg.OUT_DIR = naja::pipeline::allocate_run_dir(out_root, c.model_name);
    cfg.derive_paths();

    naja::status::phase(cfg.STATUS, "validate model contract");
    naja::status::phase(cfg.STATUS, "write generated config");
    std::string gen_cfg = cfg.OUT_DIR + "/config_generated.txt";
    naja::pipeline::write_generated_config(gen_cfg, cfg);
    cfg.source_file = make_absolute_path(gen_cfg);
    if (print_config) {
        std::cout << "config_generated    :: " << make_absolute_path(gen_cfg) << "\n";
        std::ifstream f(gen_cfg);
        if (!f.is_open()) {
            throw std::runtime_error("cannot read generated config: " + gen_cfg);
        }
        std::cout << "---\n";
        std::string line;
        while (std::getline(f, line)) std::cout << line << "\n";
        std::cout << "---\n";
    }

    {
        std::vector<std::string> argv_full;
        argv_full.reserve((size_t)argc + 3);
        argv_full.push_back("naja");
        argv_full.push_back("sample");
        argv_full.push_back("run");
        for (int i = 0; i < argc; ++i) argv_full.push_back(std::string(argv[i]));
        naja::pipeline::write_run_manifest(cfg, &c, argv_full);
    }

    naja::status::kv(cfg.STATUS, "model", c.model_name);
    naja::status::kv(cfg.STATUS, "output", make_absolute_path(cfg.OUT_DIR));
    naja::status::kv(cfg.STATUS, "bounds", (cfg.BOUNDS_FILTER ? "filter" : "ignore"));
    if (cfg.BOUNDS_FILTER) naja::status::kv(cfg.STATUS, "bounds_eps", std::to_string(cfg.BOUNDS_EPS));
    if (dry_run) {
        naja::status::phase(cfg.STATUS, "dry-run: not executing sampling");
        return;
    }

    naja::status::phase(cfg.STATUS, "sampling");
    int rc = run_sampling_job(cfg, cfg.VERBOSE, true);
    if (rc != 0) {
        throw std::runtime_error("sampling failed with return code " + std::to_string(rc));
    }
}

} // namespace naja::cli::sample


