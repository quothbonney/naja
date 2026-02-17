#include "cli/sample/commands.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli/sample/common.h"
#include "engine/job.h"
#include "pipeline/model_contract.h"
#include "pipeline/run_manifest.h"
#include "rounding/config.h"
#include "rounding/schedule_io.h"
#include "runtime_config.h"
#include "util/cli_parse.h"
#include "util/status.h"
#include "utils.h"

namespace naja::cli::sample {

namespace {

void normalize_and_validate_rounding_flags(double pair_prob,
                                           int resync_interval,
                                           int iter_rounding_passes,
                                           int iter_rounding_warmup,
                                           const std::string& pair_schedule_path) {
    naja::util::cli_parse::require_double_in_01("--pair-prob", pair_prob);
    naja::util::cli_parse::require_int_ge("--resync-interval", resync_interval, 0);
    naja::util::cli_parse::require_int_ge("--iter-rounding-passes", iter_rounding_passes, 0);
    naja::util::cli_parse::require_int_ge("--iter-rounding-warmup", iter_rounding_warmup, 0);
    naja::rounding::RoundingConfig cfg;
    cfg.pair_prob = pair_prob;
    cfg.resync_interval = resync_interval;
    cfg.iter_rounding_passes = iter_rounding_passes;
    cfg.iter_rounding_warmup = iter_rounding_warmup;
    cfg.pair_schedule_path = pair_schedule_path;
    naja::rounding::validate_rounding_config(cfg);
}

} // namespace

void cmd_run(int argc, char** argv) {
    std::string model_dir;
    std::string out_root;
    int gpu = 0;
    int n_chains = -1;
    int n_samples = -1;
    int tpb = 128;
    int thinning = 0;
    double pair_prob = 0.0;
    int resync_interval = 0;
    int iter_rounding_passes = 0;
    int iter_rounding_warmup = 0;
    double extra_constraint_eps = 0.0;
    double constraint_eps = 0.0;
    std::string pair_schedule_path;
    std::string start_policy = "file";
    double ksparse_prob = 0.0;
    int ksparse_k = 8;
    std::string extra_constraints_mode = "auto";
    double affine_hull_tol = 0.0;
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
        else if (a == "--thinning") thinning = std::stoi(next_arg(i, argc, argv, a));
        else if (a == "--pair-prob") pair_prob = std::stod(next_arg(i, argc, argv, a));
        else if (a == "--resync-interval") resync_interval = std::stoi(next_arg(i, argc, argv, a));
        else if (a == "--iter-rounding-passes") iter_rounding_passes = std::stoi(next_arg(i, argc, argv, a));
        else if (a == "--iter-rounding-warmup") iter_rounding_warmup = std::stoi(next_arg(i, argc, argv, a));
        else if (a == "--extra-constraint-eps") extra_constraint_eps = std::stod(next_arg(i, argc, argv, a));
        else if (a == "--constraint-eps") constraint_eps = std::stod(next_arg(i, argc, argv, a));
        else if (a == "--pair-schedule" || a == "--pair-schedule-file") pair_schedule_path = next_arg(i, argc, argv, a);
        else if (a == "--start-policy") start_policy = next_arg(i, argc, argv, a);
        else if (a == "--ksparse-prob") ksparse_prob = std::stod(next_arg(i, argc, argv, a));
        else if (a == "--ksparse-k") ksparse_k = std::stoi(next_arg(i, argc, argv, a));
        else if (a == "--extra-constraints") extra_constraints_mode = next_arg(i, argc, argv, a);
        else if (a == "--affine-hull-tol") affine_hull_tol = std::stod(next_arg(i, argc, argv, a));
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
    try {
        normalize_and_validate_rounding_flags(
            pair_prob,
            resync_interval,
            iter_rounding_passes,
            iter_rounding_warmup,
            pair_schedule_path
        );
    } catch (const std::invalid_argument& e) {
        die_usage(e.what());
    }
    if (extra_constraint_eps < 0.0) die_usage("invalid --extra-constraint-eps (must be >=0)");
    if (constraint_eps < 0.0) die_usage("invalid --constraint-eps (must be >=0)");
    if (start_policy != "file" && start_policy != "cube_center") die_usage("invalid --start-policy (file|cube_center)");
    if (ksparse_prob < 0.0 || ksparse_prob > 1.0) die_usage("invalid --ksparse-prob (must be in [0,1])");
    if (ksparse_k <= 0) die_usage("invalid --ksparse-k (must be >0)");
    if (affine_hull_tol < 0.0) die_usage("invalid --affine-hull-tol (must be >=0)");
    if (extra_constraints_mode != "auto" && extra_constraints_mode != "ignore" && extra_constraints_mode != "require") {
        die_usage("invalid --extra-constraints (auto|ignore|require)");
    }
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
    cfg.THINNING = thinning;
    cfg.PAIR_PROB = pair_prob;
    cfg.RESYNC_INTERVAL = resync_interval;
    cfg.ITER_ROUNDING_PASSES = iter_rounding_passes;
    cfg.ITER_ROUNDING_WARMUP = iter_rounding_warmup;
    cfg.EXTRA_CONSTRAINT_EPS = extra_constraint_eps;
    cfg.CONSTRAINT_EPS = constraint_eps;
    cfg.START_POLICY = start_policy;
    cfg.KSPARSE_PROB = ksparse_prob;
    cfg.KSPARSE_K = ksparse_k;
    cfg.EXTRA_CONSTRAINTS = extra_constraints_mode;
    cfg.AFFINE_HULL_TOL = affine_hull_tol;

    if (!pair_schedule_path.empty()) {
        // Validate early so a bad schedule doesn't waste GPU time.
        (void)naja::rounding::load_pair_schedule_csv(pair_schedule_path);
        cfg.PAIR_SCHEDULE = pair_schedule_path;
    }

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


