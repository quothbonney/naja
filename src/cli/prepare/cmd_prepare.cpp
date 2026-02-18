#include "cli/sample/commands.h"

#include <Eigen/Dense>
#include <fstream>
#include <cstdlib>
#include <atomic>
#include <exception>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sstream>
#include <thread>
#include <vector>
#include <algorithm>
#include <cmath>

#include "cli/sample/common.h"
#include "rounding/inherit.h"
#include "csv_loader.h"
#include "pipeline/feasible_start_files.h"
#include "pipeline/model_contract.h"
#include "pipeline/text_io.h"
#include "runtime_config.h"
#include "utils.h"

namespace naja::cli::sample {

namespace {

bool rounding_present_enough(const naja::pipeline::ModelContract& c) {
    const std::string r = c.rounding_dir;
    if (!is_directory(r)) return false;
    const std::string base = r + "/" + c.model_name + "_rounding_";
    if (!path_exists(base + "A.csv")) return false;
    if (!path_exists(base + "b.csv")) return false;
    if (!path_exists(base + "start.csv")) return false;
    if (!path_exists(base + "T.csv")) return false;
    if (!path_exists(base + "shift.csv")) return false;
    return true;
}

std::string hash_string(const std::string& s) {
    return std::to_string(std::hash<std::string>{}(s));
}

std::string signature_for_conditioning(const std::string& conditioner_cmd,
                                       const std::vector<std::string>& conditioner_args,
                                       const std::string& base_reaction_ids_path,
                                       const std::string& row_id) {
    std::string sig = conditioner_cmd;
    for (const auto& a : conditioner_args) sig.append("\nARG ").append(a);
    sig.append("\nROW_ID ").append(row_id);
    sig.append("\nBASE_RXN_SHA ").append(hash_string(naja::pipeline::read_all_text(base_reaction_ids_path)));
    return sig;
}

void write_signature(const std::string& path, const std::string& sig) {
    std::ofstream f(path);
    if (!f.is_open()) throw std::runtime_error("cannot write signature: " + path);
    f << sig;
}

void ensure_conditioning_signature(const std::string& sig_path, const std::string& expected_sig, bool force) {
    if (!path_exists(sig_path)) {
        if (force) return;
        throw std::runtime_error("conditioning signature missing; use --force-conditioning to overwrite");
    }
    std::string current = naja::pipeline::read_all_text(sig_path);
    if (current != expected_sig) {
        if (force) return;
        throw std::runtime_error("conditioning signature mismatch; use --force-conditioning to overwrite");
    }
}

void validate_bounds(const Eigen::VectorXd& lb,
                     const Eigen::VectorXd& ub,
                     int expected_len) {
    if (lb.size() != ub.size()) throw std::runtime_error("l_bounds size != u_bounds size");
    if (lb.size() != expected_len) {
        throw std::runtime_error("bounds length mismatch: got " + std::to_string(lb.size()) + " expected " + std::to_string(expected_len));
    }
    for (int i = 0; i < lb.size(); ++i) {
        double l = lb[i];
        double u = ub[i];
        if (!std::isfinite(l) || !std::isfinite(u)) throw std::runtime_error("non-finite bound at index " + std::to_string(i));
        if (l > u) throw std::runtime_error("l_bounds > u_bounds at index " + std::to_string(i));
    }
}

void write_matrix_csv(const std::string& path, const Eigen::MatrixXd& M) {
    std::ofstream f(path);
    if (!f.is_open()) throw std::runtime_error("cannot write: " + path);
    for (int i = 0; i < M.rows(); ++i) {
        for (int j = 0; j < M.cols(); ++j) {
            if (j) f << ",";
            f << std::setprecision(12) << M(i, j);
        }
        f << "\n";
    }
}

void write_vector_csv(const std::string& path, const Eigen::VectorXd& v) {
    std::ofstream f(path);
    if (!f.is_open()) throw std::runtime_error("cannot write: " + path);
    for (int i = 0; i < v.size(); ++i) {
        f << std::setprecision(12) << v[i] << "\n";
    }
}

void build_extra_from_bounds(const naja::pipeline::ModelContract& c,
                             const std::string& base_round_prefix,
                             const Eigen::VectorXd& lb_base,
                             const Eigen::VectorXd& ub_base,
                             const Eigen::VectorXd& lb_new,
                             const Eigen::VectorXd& ub_new) {
    const std::string base_T_path = base_round_prefix + "_T.csv";
    const std::string base_shift_path = base_round_prefix + "_shift.csv";
    Eigen::MatrixXd T = csv::loadMatrix(base_T_path);
    Eigen::VectorXd shift = csv::loadVector(base_shift_path);
    if (T.rows() != lb_base.size()) throw std::runtime_error("T rows != base bounds length");
    if (shift.size() != T.rows()) throw std::runtime_error("shift length != T rows");

    std::vector<Eigen::VectorXd> rows;
    std::vector<double> rhs;
    double eps = 1e-9;
    for (int i = 0; i < lb_base.size(); ++i) {
        if (ub_new[i] < ub_base[i] - eps) {
            rows.push_back(T.row(i));
            rhs.push_back(ub_new[i] - shift[i]);
        }
        if (lb_new[i] > lb_base[i] + eps) {
            rows.push_back(-T.row(i));
            rhs.push_back(-lb_new[i] + shift[i]);
        }
    }

    const std::string extra_A = c.rounding_dir + "/" + c.model_name + "_rounding_extra_A.csv";
    const std::string extra_b = c.rounding_dir + "/" + c.model_name + "_rounding_extra_b.csv";

    if (rows.empty()) {
        remove_if_exists(extra_A);
        remove_if_exists(extra_b);
        return;
    }

    Eigen::MatrixXd A(rows.size(), T.cols());
    for (size_t i = 0; i < rows.size(); ++i) {
        A.row((int)i) = rows[i];
    }
    Eigen::VectorXd b(rhs.size());
    for (size_t i = 0; i < rhs.size(); ++i) b[(int)i] = rhs[i];

    write_matrix_csv(extra_A, A);
    write_vector_csv(extra_b, b);
}

size_t choose_prepare_lp_workers(size_t n_models) {
    if (n_models <= 1) return 1;
    unsigned hw = std::thread::hardware_concurrency();
    size_t workers = hw ? static_cast<size_t>(hw) : 8;
    // Keep concurrency moderate by default to avoid oversubscribing memory
    // and licenses while still materially reducing wall-clock time.
    workers = std::min<size_t>(workers, 16);
    workers = std::min(workers, n_models);
    return std::max<size_t>(workers, 2);
}

void ensure_feasible_starts_parallel(const std::vector<naja::pipeline::ModelContract>& contracts) {
    if (contracts.empty()) return;

    const size_t n_workers = choose_prepare_lp_workers(contracts.size());
    if (n_workers == 1) {
        for (const auto& c : contracts) {
            naja::pipeline::ensure_feasible_rounding_start_if_extra_present(c);
            naja::pipeline::validate_contract(c, true);
        }
        return;
    }

    std::atomic<size_t> next_idx{0};
    std::exception_ptr first_error;
    std::mutex error_mutex;

    auto worker = [&]() {
        while (true) {
            const size_t idx = next_idx.fetch_add(1);
            if (idx >= contracts.size()) break;
            try {
                const auto& c = contracts[idx];
                naja::pipeline::ensure_feasible_rounding_start_if_extra_present(c);
                naja::pipeline::validate_contract(c, true);
            } catch (...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!first_error) first_error = std::current_exception();
            }
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(n_workers);
    for (size_t i = 0; i < n_workers; ++i) {
        workers.emplace_back(worker);
    }
    for (auto& t : workers) {
        t.join();
    }

    if (first_error) std::rethrow_exception(first_error);
}

} // namespace

void cmd_prepare(int argc, char** argv) {
    std::string models_root;
    std::string model_list;
    std::string base_model_dir;
    std::string mode = "symlink";
    std::string out_model_list;
    std::string out_bulk_config;
    bool dry_run = false;
    std::string conditioner_cmd;
    std::vector<std::string> conditioner_args;
    bool skip_conditioning_if_gem_present = true;
    bool force_conditioning = false;

    std::string out_dir;
    int n_chains = 4;
    int n_samples = 5000;
    int tpb = 128;
    std::string gpus;
    int gpu_device = 0;
    std::string bounds_policy = "filter";
    double bounds_eps = 1e-6;
    bool write_npy = true;
    bool write_samples_valid = false;
    bool verbose = false;

    for (int i = 0; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--models-root") models_root = next_arg(i, argc, argv, a);
        else if (a == "--model-list") model_list = next_arg(i, argc, argv, a);
        else if (a == "--base-model-dir") base_model_dir = next_arg(i, argc, argv, a);
        else if (a == "--mode") mode = next_arg(i, argc, argv, a);
        else if (a == "--out-model-list") out_model_list = next_arg(i, argc, argv, a);
        else if (a == "--out-bulk-config") out_bulk_config = next_arg(i, argc, argv, a);
        else if (a == "--dry-run") dry_run = true;
        else if (a == "--conditioner-cmd") conditioner_cmd = next_arg(i, argc, argv, a);
        else if (a == "--conditioner-arg") conditioner_args.push_back(next_arg(i, argc, argv, a));
        else if (a == "--skip-conditioning-if-gem-present") skip_conditioning_if_gem_present = true;
        else if (a == "--force-conditioning") force_conditioning = true;

        else if (a == "--out-dir") out_dir = next_arg(i, argc, argv, a);
        else if (a == "--n-chains") n_chains = std::stoi(next_arg(i, argc, argv, a));
        else if (a == "--n-samples") n_samples = std::stoi(next_arg(i, argc, argv, a));
        else if (a == "--tpb") tpb = std::stoi(next_arg(i, argc, argv, a));
        else if (a == "--gpus") gpus = next_arg(i, argc, argv, a);
        else if (a == "--gpu-device") gpu_device = std::stoi(next_arg(i, argc, argv, a));
        else if (a == "--bounds-policy") bounds_policy = next_arg(i, argc, argv, a);
        else if (a == "--bounds-eps") bounds_eps = std::stod(next_arg(i, argc, argv, a));
        else if (a == "--write-npy") write_npy = true;
        else if (a == "--no-write-npy") write_npy = false;
        else if (a == "--write-samples-valid") write_samples_valid = true;
        else if (a == "--verbose") verbose = true;
        else die_usage("unknown flag: " + a);
    }

    if (models_root.empty()) die_usage("missing --models-root");
    if (model_list.empty()) die_usage("missing --model-list");
    if (!base_model_dir.empty() && (mode != "symlink" && mode != "copy")) die_usage("invalid --mode: " + mode);
    if (!is_directory(models_root)) throw std::runtime_error("models-root is not a directory: " + models_root);

    std::vector<std::string> models = naja::pipeline::load_model_list(model_list);
    bool do_inherit = !base_model_dir.empty();

    naja::pipeline::ModelContract base;
    if (do_inherit) {
        base = naja::pipeline::parse_model_dir(base_model_dir);
        naja::pipeline::validate_contract(base, true);
    }

    if (!conditioner_cmd.empty() && base_model_dir.empty()) {
        die_usage("--conditioner-cmd requires --base-model-dir");
    }

    std::vector<std::string> prepared;
    prepared.reserve(models.size());
    std::vector<naja::pipeline::ModelContract> contracts_for_feasible_start;
    contracts_for_feasible_start.reserve(models.size());

    for (const auto& name : models) {
        naja::pipeline::ModelContract c = naja::pipeline::parse_model_dir(models_root + "/" + name);

        const std::string rxn = c.gem_dir + "/reaction_ids.txt";
        const std::string lb = c.gem_dir + "/l_bounds.csv";
        const std::string ub = c.gem_dir + "/u_bounds.csv";
        bool gem_present = is_directory(c.gem_dir) && path_exists(rxn) && path_exists(lb) && path_exists(ub);

        if (!conditioner_cmd.empty()) {
            std::string sig_expected = signature_for_conditioning(conditioner_cmd, conditioner_args, base_model_dir + "/gem/reaction_ids.txt", c.model_name);
            const std::string sig_path = c.gem_dir + "/conditioning.signature";
            bool need_condition = force_conditioning || !gem_present || !skip_conditioning_if_gem_present;
            if (gem_present && skip_conditioning_if_gem_present && !force_conditioning) {
                if (path_exists(sig_path)) {
                    ensure_conditioning_signature(sig_path, sig_expected, false);
                    need_condition = false;
                } else {
                    // We cannot prove provenance; re-run conditioner to establish signature.
                    need_condition = true;
                }
            }

            if (dry_run) {
                prepared.push_back(name);
                continue;
            }

            if (need_condition && !dry_run) {
                ensure_dir(c.gem_dir);
                std::string cmd = conditioner_cmd;
                cmd.append(" --base-model-dir ").append(base_model_dir);
                cmd.append(" --out-model-dir ").append(c.model_dir);
                cmd.append(" --row-id ").append(c.model_name);
                for (const auto& a : conditioner_args) {
                    cmd.push_back(' ');
                    cmd.append(a);
                }
                int rc = std::system(cmd.c_str());
                if (rc != 0) {
                    throw std::runtime_error("conditioner failed with exit code " + std::to_string(rc));
                }
            }

            if (!path_exists(rxn) || !path_exists(lb) || !path_exists(ub)) {
                throw std::runtime_error("conditioner did not produce required gem files for " + c.model_dir);
            }

            auto base_rxn_lines = naja::pipeline::read_nonempty_trimmed_lines(base_model_dir + "/gem/reaction_ids.txt");
            auto new_rxn_lines = naja::pipeline::read_nonempty_trimmed_lines(rxn);
            if (base_rxn_lines != new_rxn_lines) {
                throw std::runtime_error("reaction_ids mismatch vs base for " + c.model_dir);
            }

            Eigen::VectorXd lb_vec = csv::loadVector(lb);
            Eigen::VectorXd ub_vec = csv::loadVector(ub);
            validate_bounds(lb_vec, ub_vec, (int)base_rxn_lines.size());

            Eigen::VectorXd lb_base = csv::loadVector(base_model_dir + "/gem/l_bounds.csv");
            Eigen::VectorXd ub_base = csv::loadVector(base_model_dir + "/gem/u_bounds.csv");
            validate_bounds(lb_base, ub_base, lb_vec.size());

            if (!dry_run) {
                ensure_dir(c.rounding_dir);
                std::string base_round_prefix = base_model_dir + "/rounding/" + base.model_name + "_rounding";
                build_extra_from_bounds(c, base_round_prefix, lb_base, ub_base, lb_vec, ub_vec);
                write_signature(sig_path, sig_expected);
            }
        } else {
            if (!gem_present) {
                throw std::runtime_error("gem/ missing and no conditioner provided for " + c.model_dir);
            }
        }

        if (!dry_run) {
            ensure_dir(c.rounding_dir);
            naja::rounding::normalize_extra_constraints(c);
        }

        if (!rounding_present_enough(c)) {
            if (!do_inherit) {
                throw std::runtime_error("rounding contract missing/invalid and no --base-model-dir provided: " + c.model_dir);
            }
            if (!dry_run) {
                naja::rounding::inherit_rounding_impl(base, c, mode);
                naja::rounding::normalize_extra_constraints(c);
            }
            naja::pipeline::validate_contract(c, true);
        } else {
            naja::pipeline::validate_contract(c, true);
        }

        if (!dry_run) {
            contracts_for_feasible_start.push_back(c);
        }
        prepared.push_back(name);
    }

    if (!dry_run) {
        ensure_feasible_starts_parallel(contracts_for_feasible_start);
    }

    if (!out_model_list.empty() && !dry_run) {
        std::ofstream f(out_model_list);
        if (!f.is_open()) throw std::runtime_error("cannot write out-model-list: " + out_model_list);
        for (const auto& m : prepared) f << m << "\n";
    }

    if (!out_bulk_config.empty()) {
        if (out_model_list.empty()) die_usage("--out-bulk-config requires --out-model-list");
        if (out_dir.empty()) die_usage("--out-bulk-config requires --out-dir");
        if (!dry_run) {
            RuntimeConfig cfg;
            cfg.DATA_DIR = models_root;
            cfg.MODEL_NAME = "bulk";
            cfg.OUT_DIR = out_dir;
            cfg.N_CHAINS = n_chains;
            cfg.N_SAMPLES = n_samples;
            cfg.TPB_SS = tpb;
            cfg.GPU_DEVICE = gpu_device;
            cfg.GPU_LIST = gpus;
            cfg.BACK_TRANSFORM = true;
            cfg.WRITE_DATA = write_npy;
            cfg.VERBOSE = verbose;
            cfg.STATUS = true;
            cfg.BOUNDS_FILTER = (bounds_policy == "filter");
            cfg.BOUNDS_EPS = bounds_eps;
            cfg.WRITE_SAMPLES_VALID = write_samples_valid;
            cfg.BULK_MODEL_LIST = out_model_list;
            cfg.SKIP_EXISTING = false;
            cfg.derive_paths();

            naja::pipeline::write_generated_config(out_bulk_config, cfg);
        }
    }

    std::cout << "OK\n";
    std::cout << "models_root     :: " << make_absolute_path(models_root) << "\n";
    std::cout << "model_list      :: " << make_absolute_path(model_list) << "\n";
    std::cout << "prepared        :: " << prepared.size() << " / " << models.size() << "\n";
    if (!out_model_list.empty()) std::cout << "out_model_list  :: " << make_absolute_path(out_model_list) << "\n";
    if (!out_bulk_config.empty()) std::cout << "out_bulk_config :: " << make_absolute_path(out_bulk_config) << "\n";
    if (dry_run) std::cout << "dry_run         :: true\n";
}

} // namespace naja::cli::sample
