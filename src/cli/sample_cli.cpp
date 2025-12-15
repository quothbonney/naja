#include "cli/sample_cli.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "job.h"
#include "pipeline/config_io.h"
#include "pipeline/model_contract.h"
#include "pipeline/run_manifest.h"
#include "pipeline/run_layout.h"
#include "pipeline/verify_report.h"
#include "runtime_config.h"
#include "util/status.h"
#include "utils.h"

#include <fstream>
#include <sys/stat.h>
#include <unistd.h>

namespace {

[[noreturn]] void die_usage(const std::string& msg) {
    std::cerr << "error: " << msg << "\n\n";
    std::cerr << "usage:\n";
    std::cerr << "  naja sample run --model-dir <dir> --out-root <dir> --gpu <id> --n-chains <n> --n-samples <n> [--tpb <n>] [--backmap] [--write-npy] [--bounds-policy ignore|filter] [--bounds-eps <eps>] [--write-samples-valid] [--verbose] [--quiet] [--dry-run] [--print-config]\n";
    std::cerr << "  naja sample bulk --models-root <dir> --model-list <file> --out-root <dir> --name <runname> --gpus <csv> --n-chains <n> --n-samples <n> [--tpb <n>] [--backmap] [--write-npy] [--bounds-policy ignore|filter] [--bounds-eps <eps>] [--write-samples-valid] [--verbose] [--quiet] [--dry-run] [--print-config]\n";
    std::cerr << "  naja sample verify --model-dir <dir> [--backmap]\n";
    std::cerr << "  naja sample inherit-rounding --base-model-dir <dir> --target-model-dir <dir> [--mode symlink|copy]\n";
    std::cerr << "  naja sample prepare --models-root <dir> --model-list <file> [--base-model-dir <dir> --mode symlink|copy] [--out-model-list <file>] [--dry-run]\n";
    std::exit(2);
}

std::string next_arg(int& i, int argc, char** argv, const std::string& flag) {
    if (i + 1 >= argc) {
        die_usage("missing value for " + flag);
    }
    return std::string(argv[++i]);
}

void require_nonempty_file(const std::string& path, const std::string& what) {
    if (!path_exists(path)) {
        throw std::runtime_error("missing required " + what + ": " + path);
    }
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        throw std::runtime_error("cannot stat required " + what + ": " + path);
    }
    if (st.st_size == 0) {
        throw std::runtime_error("empty " + what + " is illegal: " + path);
    }
}

bool file_is_empty(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        throw std::runtime_error("cannot stat: " + path);
    }
    return st.st_size == 0;
}

void remove_if_exists(const std::string& path) {
    if (path_exists(path)) {
        if (unlink(path.c_str()) != 0) {
            throw std::runtime_error("cannot remove: " + path);
        }
    }
}

void normalize_extra_constraints(const naja::pipeline::ModelContract& c) {
    const std::string extra_A = c.rounding_dir + "/" + c.model_name + "_rounding_extra_A.csv";
    const std::string extra_b = c.rounding_dir + "/" + c.model_name + "_rounding_extra_b.csv";
    bool ha = path_exists(extra_A);
    bool hb = path_exists(extra_b);
    if (ha != hb) {
        throw std::runtime_error("extra constraints must be both-present or both-absent: " + extra_A + " / " + extra_b);
    }
    if (!ha) {
        return;
    }
    // If both exist but either is empty, treat as "no extra constraints": remove both.
    if (file_is_empty(extra_A) || file_is_empty(extra_b)) {
        remove_if_exists(extra_A);
        remove_if_exists(extra_b);
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
        require_nonempty_file(src, std::string("base rounding ") + suf);

        remove_if_exists(dst);

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

void cmd_verify(int argc, char** argv) {
    std::string model_dir;
    bool backmap = false;

    for (int i = 0; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model-dir") model_dir = next_arg(i, argc, argv, a);
        else if (a == "--backmap") backmap = true;
        else die_usage("unknown flag: " + a);
    }
    if (model_dir.empty()) die_usage("missing --model-dir");

    naja::pipeline::ModelContract c = naja::pipeline::parse_model_dir(model_dir);
    naja::pipeline::validate_contract(c, backmap);

    const std::string A_path = c.rounding_dir + "/" + c.model_name + "_rounding_A.csv";
    const std::string b_path = c.rounding_dir + "/" + c.model_name + "_rounding_b.csv";
    const std::string start_path = c.rounding_dir + "/" + c.model_name + "_rounding_start.csv";
    const std::string T_path = c.rounding_dir + "/" + c.model_name + "_rounding_T.csv";
    const std::string shift_path = c.rounding_dir + "/" + c.model_name + "_rounding_shift.csv";
    const std::string extra_A_path = c.rounding_dir + "/" + c.model_name + "_rounding_extra_A.csv";
    const std::string extra_b_path = c.rounding_dir + "/" + c.model_name + "_rounding_extra_b.csv";

    naja::pipeline::CsvShape A = naja::pipeline::csv_shape(A_path);
    naja::pipeline::CsvShape b = naja::pipeline::csv_shape(b_path);
    naja::pipeline::CsvShape start = naja::pipeline::csv_shape(start_path);
    if (b.rows != A.rows) {
        throw std::runtime_error("b rows != A rows");
    }
    if (start.rows != A.cols) {
        throw std::runtime_error("start dim != A cols");
    }

    int extra_rows = 0;
    bool extra_present = path_exists(extra_A_path);
    if (extra_present) {
        naja::pipeline::CsvShape extraA = naja::pipeline::csv_shape(extra_A_path);
        naja::pipeline::CsvShape extrab = naja::pipeline::csv_shape(extra_b_path);
        extra_rows = extraA.rows;
        if (extraA.rows != extrab.rows) {
            throw std::runtime_error("extra_A rows != extra_b rows");
        }
        if (extraA.cols != A.cols) {
            throw std::runtime_error("extra_A cols != A cols");
        }
    }

    int rxn_ids_n = 0;
    int lb_n = 0;
    int ub_n = 0;
    bool gem_ok = false;
    bool gem_exists = is_directory(c.gem_dir);
    if (gem_exists) {
        const std::string rxn_ids = c.gem_dir + "/reaction_ids.txt";
        const std::string lb = c.gem_dir + "/l_bounds.csv";
        const std::string ub = c.gem_dir + "/u_bounds.csv";
        if (path_exists(rxn_ids) && path_exists(lb) && path_exists(ub)) {
            rxn_ids_n = naja::pipeline::text_line_count(rxn_ids);
            lb_n = naja::pipeline::csv_shape(lb).rows;
            ub_n = naja::pipeline::csv_shape(ub).rows;
            gem_ok = (rxn_ids_n == lb_n) && (rxn_ids_n == ub_n);
        }
    }

    std::cout << "OK\n";
    std::cout << "model_dir           :: " << c.model_dir << "\n";
    std::cout << "model_name          :: " << c.model_name << "\n";
    std::cout << "A                   :: " << A.rows << " x " << A.cols << "\n";
    std::cout << "b                   :: " << b.rows << " x " << b.cols << "\n";
    std::cout << "start               :: " << start.rows << " x " << start.cols << "\n";
    if (backmap) {
        naja::pipeline::CsvShape T = naja::pipeline::csv_shape(T_path);
        naja::pipeline::CsvShape shift = naja::pipeline::csv_shape(shift_path);
        if (T.cols != A.cols) {
            throw std::runtime_error("T cols != A cols");
        }
        if (shift.rows != T.rows) {
            throw std::runtime_error("shift dim != T rows");
        }
        std::cout << "T                   :: " << T.rows << " x " << T.cols << "\n";
        std::cout << "shift               :: " << shift.rows << " x " << shift.cols << "\n";
    }
    std::cout << "extra_constraints   :: " << (extra_present ? "present" : "absent") << "\n";
    if (extra_present) {
        std::cout << "extra_rows          :: " << extra_rows << "\n";
    }
    std::cout << "gem_dir             :: " << (gem_exists ? "present" : "absent") << "\n";
    if (gem_exists) {
        std::cout << "gem_counts          :: rxn_ids=" << rxn_ids_n << " lb=" << lb_n << " ub=" << ub_n << "\n";
        std::cout << "gem_consistent      :: " << (gem_ok ? "yes" : "no") << "\n";
    }
}

void cmd_inherit_rounding(int argc, char** argv) {
    std::string base_model_dir;
    std::string target_model_dir;
    std::string mode = "symlink";

    for (int i = 0; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--base-model-dir") base_model_dir = next_arg(i, argc, argv, a);
        else if (a == "--target-model-dir") target_model_dir = next_arg(i, argc, argv, a);
        else if (a == "--mode") mode = next_arg(i, argc, argv, a);
        else die_usage("unknown flag: " + a);
    }
    if (base_model_dir.empty()) die_usage("missing --base-model-dir");
    if (target_model_dir.empty()) die_usage("missing --target-model-dir");
    if (mode != "symlink" && mode != "copy") die_usage("invalid --mode: " + mode);

    naja::pipeline::ModelContract base = naja::pipeline::parse_model_dir(base_model_dir);
    naja::pipeline::ModelContract target = naja::pipeline::parse_model_dir(target_model_dir);
    inherit_rounding_impl(base, target, mode);
    std::cout << "OK\n";
    std::cout << "base_model_dir   :: " << base.model_dir << "\n";
    std::cout << "target_model_dir :: " << target.model_dir << "\n";
}

void cmd_prepare(int argc, char** argv) {
    std::string models_root;
    std::string model_list;
    std::string base_model_dir;
    std::string mode = "symlink";
    std::string out_model_list;
    bool dry_run = false;

    for (int i = 0; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--models-root") models_root = next_arg(i, argc, argv, a);
        else if (a == "--model-list") model_list = next_arg(i, argc, argv, a);
        else if (a == "--base-model-dir") base_model_dir = next_arg(i, argc, argv, a);
        else if (a == "--mode") mode = next_arg(i, argc, argv, a);
        else if (a == "--out-model-list") out_model_list = next_arg(i, argc, argv, a);
        else if (a == "--dry-run") dry_run = true;
        else die_usage("unknown flag: " + a);
    }

    if (models_root.empty()) die_usage("missing --models-root");
    if (model_list.empty()) die_usage("missing --model-list");
    if (!base_model_dir.empty() && (mode != "symlink" && mode != "copy")) die_usage("invalid --mode: " + mode);
    if (!is_directory(models_root)) {
        throw std::runtime_error("models-root is not a directory: " + models_root);
    }

    std::vector<std::string> models = naja::pipeline::load_model_list(model_list);
    bool do_inherit = !base_model_dir.empty();
    naja::pipeline::ModelContract base;
    if (do_inherit) {
        base = naja::pipeline::parse_model_dir(base_model_dir);
        // Base must support backmap for bounds-policy=filter workflows.
        naja::pipeline::validate_contract(base, true);
    }

    std::vector<std::string> prepared;
    prepared.reserve(models.size());

    int inherited = 0;
    int skipped_missing_gem = 0;
    int failed = 0;

    for (const auto& name : models) {
        try {
            naja::pipeline::ModelContract c = naja::pipeline::parse_model_dir(models_root + "/" + name);

            // Require GEM exports for the typical workflow.
            const std::string rxn = c.gem_dir + "/reaction_ids.txt";
            const std::string lb = c.gem_dir + "/l_bounds.csv";
            const std::string ub = c.gem_dir + "/u_bounds.csv";
            if (!(is_directory(c.gem_dir) && path_exists(rxn) && path_exists(lb) && path_exists(ub))) {
                skipped_missing_gem++;
                continue;
            }

            if (!dry_run) {
                ensure_dir(c.rounding_dir);
                normalize_extra_constraints(c);
            }

            bool ok = true;
            try {
                naja::pipeline::validate_contract(c, true);
            } catch (const std::exception&) {
                ok = false;
            }

            if (!ok) {
                if (!do_inherit) {
                    throw std::runtime_error("rounding contract missing/invalid and no --base-model-dir provided: " + c.model_dir);
                }
                if (!dry_run) {
                    inherit_rounding_impl(base, c, mode);
                    normalize_extra_constraints(c);
                }
                inherited++;
            }

            naja::pipeline::validate_contract(c, true);
            prepared.push_back(name);
        } catch (const std::exception& e) {
            failed++;
            std::cerr << "prepare failed: " << name << " :: " << e.what() << "\n";
        }
    }

    if (!out_model_list.empty() && !dry_run) {
        std::ofstream f(out_model_list);
        if (!f.is_open()) {
            throw std::runtime_error("cannot write out-model-list: " + out_model_list);
        }
        for (const auto& m : prepared) f << m << "\n";
    }

    std::cout << "OK\n";
    std::cout << "models_root         :: " << make_absolute_path(models_root) << "\n";
    std::cout << "model_list          :: " << make_absolute_path(model_list) << "\n";
    std::cout << "prepared            :: " << prepared.size() << " / " << models.size() << "\n";
    std::cout << "inherited           :: " << inherited << "\n";
    std::cout << "skipped_missing_gem :: " << skipped_missing_gem << "\n";
    std::cout << "failed              :: " << failed << "\n";
    if (!out_model_list.empty()) {
        std::cout << "out_model_list      :: " << make_absolute_path(out_model_list) << "\n";
    }
    if (dry_run) {
        std::cout << "dry_run             :: true\n";
    }
}

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
        while (std::getline(f, line)) {
            std::cout << line << "\n";
        }
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

void cmd_bulk(int argc, char** argv) {
    std::string models_root;
    std::string model_list;
    std::string out_root;
    std::string name;
    std::string gpus;
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
        if (a == "--models-root") models_root = next_arg(i, argc, argv, a);
        else if (a == "--model-list") model_list = next_arg(i, argc, argv, a);
        else if (a == "--out-root") out_root = next_arg(i, argc, argv, a);
        else if (a == "--name") name = next_arg(i, argc, argv, a);
        else if (a == "--gpus") gpus = next_arg(i, argc, argv, a);
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

    if (models_root.empty()) die_usage("missing --models-root");
    if (model_list.empty()) die_usage("missing --model-list");
    if (out_root.empty()) die_usage("missing --out-root");
    if (name.empty()) die_usage("missing --name");
    if (gpus.empty()) die_usage("missing --gpus");
    if (n_chains <= 0) die_usage("invalid --n-chains");
    if (n_samples <= 0) die_usage("invalid --n-samples");
    if (bounds_policy != "ignore" && bounds_policy != "filter") die_usage("invalid --bounds-policy: " + bounds_policy);
    if (bounds_policy == "filter" && !backmap) {
        throw std::runtime_error("bounds-policy=filter requires --backmap");
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
    cfg.OUT_DIR = naja::pipeline::allocate_run_dir(out_root, name);
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
        while (std::getline(f, line)) {
            std::cout << line << "\n";
        }
        std::cout << "---\n";
    }

    {
        std::vector<std::string> argv_full;
        argv_full.reserve((size_t)argc + 3);
        argv_full.push_back("naja");
        argv_full.push_back("sample");
        argv_full.push_back("bulk");
        for (int i = 0; i < argc; ++i) argv_full.push_back(std::string(argv[i]));
        naja::pipeline::write_run_manifest(cfg, nullptr, argv_full);
    }

    naja::status::kv(cfg.STATUS, "bulk_name", name);
    naja::status::kv(cfg.STATUS, "jobs", std::to_string(models.size()));
    naja::status::kv(cfg.STATUS, "gpus", gpus);
    naja::status::kv(cfg.STATUS, "output", make_absolute_path(cfg.OUT_DIR));
    naja::status::kv(cfg.STATUS, "bounds", (cfg.BOUNDS_FILTER ? "filter" : "ignore"));
    if (cfg.BOUNDS_FILTER) naja::status::kv(cfg.STATUS, "bounds_eps", std::to_string(cfg.BOUNDS_EPS));
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

} // namespace

int naja_sample_cli_main(int argc, char** argv) {
    if (argc < 1) die_usage("missing subcommand");
    std::string sub = argv[0];

    if (sub == "run") {
        cmd_run(argc - 1, argv + 1);
        return 0;
    }
    if (sub == "bulk") {
        cmd_bulk(argc - 1, argv + 1);
        return 0;
    }
    if (sub == "verify") {
        cmd_verify(argc - 1, argv + 1);
        return 0;
    }
    if (sub == "inherit-rounding") {
        cmd_inherit_rounding(argc - 1, argv + 1);
        return 0;
    }
    if (sub == "prepare") {
        cmd_prepare(argc - 1, argv + 1);
        return 0;
    }

    die_usage("unknown subcommand: " + sub);
}


