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
#include "runtime_config.h"
#include "util/status.h"
#include "utils.h"

namespace naja::cli::sample {

void cmd_bulk(int argc, char** argv) {
    std::string models_root;
    std::string model_list;
    std::string out_root;
    std::string resume_from;
    std::string name;
    std::string gpus;
    int n_chains = -1;
    int n_samples = -1;
    int tpb = 128;
    int thinning = 0;
    double constraint_eps = 0.0;
    double affine_hull_tol = 0.0;
    double pair_prob = 0.0;
    std::string start_policy = "file";
    std::string extra_constraints_mode = "auto";
    std::string pair_schedule_method;
    bool barrier_rotate = false;
    bool barrier_whiten = false;
    bool backmap = false;
    bool write_npy = false;
    bool verbose = false;
    std::string bounds_policy = "ignore";
    double bounds_eps = 1e-6;
    bool write_samples_valid = false;
    bool quiet = false;
    bool dry_run = false;
    bool print_config = false;
    bool skip_existing = false;
    std::string flat_output;

    for (int i = 0; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--models-root") models_root = next_arg(i, argc, argv, a);
        else if (a == "--model-list") model_list = next_arg(i, argc, argv, a);
        else if (a == "--out-root") out_root = next_arg(i, argc, argv, a);
        else if (a == "--resume-from") resume_from = next_arg(i, argc, argv, a);
        else if (a == "--name") name = next_arg(i, argc, argv, a);
        else if (a == "--gpus") gpus = next_arg(i, argc, argv, a);
        else if (a == "--n-chains") n_chains = std::stoi(next_arg(i, argc, argv, a));
        else if (a == "--n-samples") n_samples = std::stoi(next_arg(i, argc, argv, a));
        else if (a == "--tpb") tpb = std::stoi(next_arg(i, argc, argv, a));
        else if (a == "--thinning") thinning = std::stoi(next_arg(i, argc, argv, a));
        else if (a == "--constraint-eps") constraint_eps = std::stod(next_arg(i, argc, argv, a));
        else if (a == "--affine-hull-tol") affine_hull_tol = std::stod(next_arg(i, argc, argv, a));
        else if (a == "--pair-prob") pair_prob = std::stod(next_arg(i, argc, argv, a));
        else if (a == "--start-policy") start_policy = next_arg(i, argc, argv, a);
        else if (a == "--extra-constraints") extra_constraints_mode = next_arg(i, argc, argv, a);
        else if (a == "--pair-schedule-method") pair_schedule_method = next_arg(i, argc, argv, a);
        else if (a == "--barrier-rotate") barrier_rotate = true;
        else if (a == "--barrier-whiten") { barrier_whiten = true; barrier_rotate = true; }
        else if (a == "--backmap") backmap = true;
        else if (a == "--write-npy") write_npy = true;
        else if (a == "--verbose") verbose = true;
        else if (a == "--bounds-policy") bounds_policy = next_arg(i, argc, argv, a);
        else if (a == "--bounds-eps") bounds_eps = std::stod(next_arg(i, argc, argv, a));
        else if (a == "--write-samples-valid") write_samples_valid = true;
        else if (a == "--quiet") quiet = true;
        else if (a == "--dry-run") dry_run = true;
        else if (a == "--print-config") print_config = true;
        else if (a == "--skip-existing") skip_existing = true;
        else if (a == "--flat-output") flat_output = next_arg(i, argc, argv, a);
        else if (a == "--help" || a == "-h") die_usage("", 0);
        else die_usage("unknown flag: " + a);
    }

    if (models_root.empty()) die_usage("missing --models-root");
    if (model_list.empty()) die_usage("missing --model-list");
    if (name.empty()) die_usage("missing --name");
    if (gpus.empty()) die_usage("missing --gpus");
    if (n_chains <= 0) die_usage("invalid --n-chains");
    if (n_samples <= 0) die_usage("invalid --n-samples");
    if (bounds_policy != "ignore" && bounds_policy != "filter") die_usage("invalid --bounds-policy: " + bounds_policy);
    if (bounds_policy == "filter" && !backmap) {
        throw std::runtime_error("bounds-policy=filter requires --backmap");
    }
    if (pair_prob < 0.0 || pair_prob > 1.0) die_usage("invalid --pair-prob (must be in [0,1])");
    if (!pair_schedule_method.empty() && pair_schedule_method != "barrier") {
        die_usage("invalid --pair-schedule-method (barrier)");
    }

    if (!is_directory(models_root)) {
        throw std::runtime_error("models-root is not a directory: " + models_root);
    }

    std::vector<std::string> models = naja::pipeline::load_model_list(model_list);
    for (const auto& m : models) {
        naja::pipeline::ModelContract c = naja::pipeline::parse_model_dir(models_root + "/" + m);
        naja::pipeline::validate_contract(c, backmap);
    }

    RuntimeConfig cfg;
    cfg.DATA_DIR = models_root;
    cfg.MODEL_NAME = name;
    cfg.N_CHAINS = n_chains;
    cfg.N_SAMPLES = n_samples;
    cfg.TPB_SS = tpb;
    cfg.GPU_DEVICE = 0;
    cfg.BACK_TRANSFORM = backmap;
    cfg.WRITE_DATA = write_npy;
    cfg.VERBOSE = verbose;
    cfg.STATUS = !quiet;
    cfg.GPU_LIST = gpus;
    cfg.BULK_MODEL_LIST = model_list;
    cfg.BOUNDS_FILTER = (bounds_policy == "filter");
    cfg.BOUNDS_EPS = bounds_eps;
    cfg.WRITE_SAMPLES_VALID = write_samples_valid;
    cfg.SKIP_EXISTING = skip_existing;
    cfg.FLAT_OUTPUT_DIR = flat_output;
    cfg.THINNING = thinning;
    cfg.PAIR_PROB = pair_prob;
    cfg.CONSTRAINT_EPS = constraint_eps;
    cfg.AFFINE_HULL_TOL = affine_hull_tol;
    cfg.START_POLICY = start_policy;
    cfg.EXTRA_CONSTRAINTS = extra_constraints_mode;
    cfg.PAIR_SCHEDULE_METHOD = pair_schedule_method;
    cfg.BARRIER_ROTATE = barrier_rotate;
    cfg.BARRIER_WHITEN = barrier_whiten;

    if (!resume_from.empty()) {
        if (!is_directory(resume_from)) {
            throw std::runtime_error("resume-from is not a directory: " + resume_from);
        }
        cfg.OUT_DIR = resume_from;
    } else {
        if (out_root.empty()) die_usage("missing --out-root (or use --resume-from)");
        cfg.OUT_DIR = naja::pipeline::allocate_run_dir(out_root, name);
    }
    cfg.derive_paths();

    naja::status::phase(cfg.STATUS, "validate model contracts");
    naja::status::phase(cfg.STATUS, "write generated bulk config");
    std::string gen_cfg = cfg.OUT_DIR + "/bulk_config_generated.txt";
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
        argv_full.push_back("bulk");
        for (int i = 0; i < argc; ++i) argv_full.push_back(std::string(argv[i]));
        const std::string mf = (!resume_from.empty() && path_exists(cfg.OUT_DIR + "/run_manifest.json"))
            ? "run_manifest_resume.json"
            : "run_manifest.json";
        naja::pipeline::write_run_manifest(cfg, nullptr, argv_full, mf);
    }

    naja::status::kv(cfg.STATUS, "bulk_name", name);
    naja::status::kv(cfg.STATUS, "jobs", std::to_string(models.size()));
    naja::status::kv(cfg.STATUS, "gpus", gpus);
    naja::status::kv(cfg.STATUS, "output", make_absolute_path(cfg.OUT_DIR));
    if (!cfg.FLAT_OUTPUT_DIR.empty()) {
        naja::status::kv(cfg.STATUS, "flat_output", make_absolute_path(cfg.FLAT_OUTPUT_DIR));
    }
    naja::status::kv(cfg.STATUS, "bounds", (cfg.BOUNDS_FILTER ? "filter" : "ignore"));
    if (cfg.BOUNDS_FILTER) naja::status::kv(cfg.STATUS, "bounds_eps", std::to_string(cfg.BOUNDS_EPS));
    if (cfg.SKIP_EXISTING) naja::status::kv(cfg.STATUS, "skip_existing", "true");

    if (dry_run) {
        naja::status::phase(cfg.STATUS, "dry-run: not executing bulk sampling");
        return;
    }

    naja::status::phase(cfg.STATUS, "bulk sampling");
    int rc = run_bulk_mode(cfg);
    if (rc != 0) {
        throw std::runtime_error("bulk sampling failed with return code " + std::to_string(rc));
    }
}

} // namespace naja::cli::sample


