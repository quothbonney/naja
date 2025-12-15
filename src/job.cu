#include "job.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <Eigen/Dense>

#include "dmatrix.h"
#include "dvector.h"
#include "gpusamplers.h"
#include "device_utils.h"
#include "csv_loader.h"
#include "npy.h"
#include "profile.h"
#include "utils.h"

using namespace naja::gpu;

// Private helper for unique job output directories
namespace {
    std::string trim_copy(const std::string& input) {
        const char* whitespace = " \t\r\n";
        size_t start = input.find_first_not_of(whitespace);
        if (start == std::string::npos) return "";
        size_t end = input.find_last_not_of(whitespace);
        return input.substr(start, end - start + 1);
    }

    std::vector<std::string> load_bulk_jobs(const RuntimeConfig& cfg) {
        std::vector<std::string> jobs;
        if (cfg.BULK_MODEL_LIST.empty()) {
            return jobs;
        }
        std::string list_path = make_absolute_path(cfg.BULK_MODEL_LIST);
        std::ifstream file(list_path);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open bulk model list: " + list_path);
        }
        std::string line;
        while (std::getline(file, line)) {
            line = trim_copy(line);
            if (line.empty() || line[0] == '#') continue;
            jobs.push_back(line);
        }
        return jobs;
    }

    std::vector<int> resolve_gpu_ids(const RuntimeConfig& cfg) {
        std::vector<int> ids;
        if (!cfg.GPU_LIST.empty()) {
            std::stringstream ss(cfg.GPU_LIST);
            std::string token;
            while (std::getline(ss, token, ',')) {
                token = trim_copy(token);
                if (token.empty()) continue;
                try {
                    ids.push_back(std::stoi(token));
                } catch (const std::exception&) {
                    throw std::runtime_error("Invalid GPU id in GPU_LIST: " + token);
                }
            }
        }
        if (ids.empty()) {
            ids.push_back(cfg.GPU_DEVICE);
        }
        return ids;
    }

    std::string make_job_output_dir(const std::string& bulk_root, const std::string& model_name) {
        std::string root = bulk_root.empty() ? "./out" : bulk_root;
        ensure_dir(root);
        std::string model_root = root + "/" + model_name;
        ensure_dir(model_root);
        std::string date = current_datestamp_compact();
        int idx = 1;
        while (true) {
            std::ostringstream oss;
            oss << model_root << "/" << model_name << "_" << date << "_" << std::setw(3) << std::setfill('0') << idx;
            std::string candidate = oss.str();
            if (!path_exists(candidate)) {
                ensure_dir(candidate);
                return candidate;
            }
            ++idx;
        }
    }

    void write_config_snapshot(const RuntimeConfig& cfg, const ProfileData& profile, const std::string& timestamp) {
        const std::string dest = cfg.OUT_DIR + "/config_used.txt";
        std::ifstream src(cfg.source_file);
        std::ofstream dst(dest);
        if (!dst.is_open()) {
            std::cerr << "WARNING: Could not write config snapshot to " << dest << std::endl;
            return;
        }

        dst << "# Source config: " << cfg.source_file << "\n";
        if (!timestamp.empty()) {
            dst << "# Captured at: " << timestamp << "\n";
        }
        dst << "# Output directory: " << make_absolute_path(cfg.OUT_DIR) << "\n\n";

        if (src.is_open()) {
            dst << src.rdbuf();
            src.close();
        } else {
            dst << "# WARNING: original config file could not be read at runtime.\n";
        }

        dst << "\n# Derived metadata\n";
        dst << "MODEL_NAME=" << cfg.MODEL_NAME << "\n";
        dst << "DATA_DIR_ABS=" << make_absolute_path(cfg.DATA_DIR) << "\n";
        dst << "MODEL_DIR_ABS=" << make_absolute_path(cfg.MODEL_DIR) << "\n";
        dst << "ROUND_PREFIX_ABS=" << make_absolute_path(cfg.ROUND_PREFIX) << "\n";
        dst << "A_FILE=" << make_absolute_path(cfg.A_FILE) << "\n";
        dst << "B_FILE=" << make_absolute_path(cfg.B_FILE) << "\n";
        dst << "START_FILE=" << make_absolute_path(cfg.START_FILE) << "\n";
        dst << "T_FILE=" << make_absolute_path(cfg.T_FILE) << "\n";
        dst << "SHIFT_FILE=" << make_absolute_path(cfg.SHIFT_FILE) << "\n";
        dst << "NPY_FILE=" << make_absolute_path(cfg.NPY_FILE) << "\n";
        dst << "PROFILE_FILE=" << make_absolute_path(cfg.PROFILE_FILE) << "\n";
        dst << "N_CHAINS=" << profile.n_chains << "\n";
        dst << "N_SAMPLES=" << profile.n_samples << "\n";
        dst << "THINNING=" << profile.thinning << "\n";
        dst << "GPU_DEVICE=" << cfg.GPU_DEVICE << "\n";
        dst << "BACK_TRANSFORM=" << (cfg.BACK_TRANSFORM ? "true" : "false") << "\n";
        dst << "WRITE_DATA=" << (cfg.WRITE_DATA ? "true" : "false") << "\n";
        dst << std::flush;
    }
}

int run_sampling_job(RuntimeConfig cfg, bool verbose, bool show_device_banner) {
    ensure_dir(cfg.OUT_DIR);

    Timer total_timer;
    ProfileData profile;

    set_device(cfg.GPU_DEVICE);
    if (show_device_banner && verbose) {
        auto devices = list_devices();
        std::cout << "devices" << std::endl;
        for (const auto& dev : devices) {
            std::cout << "  " << dev << std::endl;
        }
        std::cout << std::endl;
    }

    if (verbose) {
        std::cout << "config" << std::endl;
        std::cout << "  model           " << cfg.MODEL_NAME << std::endl;
        std::cout << "  gpu_device      " << cfg.GPU_DEVICE << std::endl;
        std::cout << "  n_chains        " << cfg.N_CHAINS << std::endl;
        std::cout << "  n_samples       " << cfg.N_SAMPLES << std::endl;
        std::cout << "  tpb_ss          " << cfg.TPB_SS << std::endl;
        std::cout << "  back_transform  " << (cfg.BACK_TRANSFORM ? "yes" : "no") << std::endl;
        std::cout << std::endl;
    }

    if (verbose) std::cout << "> loading polytope" << std::endl;
    Timer load_timer;
    Eigen::MatrixXd A_host = csv::loadMatrix(cfg.A_FILE);
    Eigen::VectorXd b_host = csv::loadVector(cfg.B_FILE);
    Eigen::VectorXd x0_host = csv::loadVector(cfg.START_FILE);
    bool has_extra_A = path_exists(cfg.A_EXTRA_FILE);
    bool has_extra_b = path_exists(cfg.B_EXTRA_FILE);
    if (has_extra_A || has_extra_b) {
        if (!has_extra_A || !has_extra_b) {
            throw std::runtime_error("incomplete extra constraint files (need both A_extra and b_extra)");
        }
        Eigen::MatrixXd A_extra = csv::loadMatrix(cfg.A_EXTRA_FILE);
        Eigen::VectorXd b_extra = csv::loadVector(cfg.B_EXTRA_FILE);
        if (A_extra.rows() != b_extra.size()) {
            throw std::runtime_error("extra constraint matrices have mismatched dimensions");
        }
        if (A_extra.cols() != A_host.cols()) {
            throw std::runtime_error("extra constraint matrix has wrong number of columns");
        }
        Eigen::MatrixXd A_aug(A_host.rows() + A_extra.rows(), A_host.cols());
        A_aug.topRows(A_host.rows()) = A_host;
        A_aug.bottomRows(A_extra.rows()) = A_extra;
        Eigen::VectorXd b_aug(b_host.size() + b_extra.size());
        b_aug.head(b_host.size()) = b_host;
        b_aug.tail(b_extra.size()) = b_extra;
        A_host = std::move(A_aug);
        b_host = std::move(b_aug);
        if (verbose) {
            std::cout << "  + extra constraints : " << b_extra.size() << std::endl;
        }
    }
    profile.load_time = load_timer.elapsed();

    int m = A_host.rows();
    int n = A_host.cols();

    if (x0_host.size() != n) {
        throw std::runtime_error("dimension mismatch: start point size (" + std::to_string(x0_host.size()) + ") != cols (" + std::to_string(n) + ")");
    }
    if (b_host.size() != m) {
        throw std::runtime_error("dimension mismatch: rhs vector size (" + std::to_string(b_host.size()) + ") != rows (" + std::to_string(m) + ")");
    }

    int thinning = std::max(n / 6, 1);
    if (verbose) {
        std::cout << "  constraints     " << m << std::endl;
        std::cout << "  reduced_dim     " << n << std::endl;
        std::cout << "  thinning        " << thinning << std::endl;
        std::cout << "  time            " << std::fixed << std::setprecision(3) << profile.load_time << "s" << std::endl;
        std::cout << std::endl;
    }

    profile.n_chains = cfg.N_CHAINS;
    profile.n_samples = cfg.N_SAMPLES;
    profile.reduced_dim = n;
    profile.constraints = m;
    profile.thinning = thinning;

    if (verbose) std::cout << "> uploading to gpu" << std::endl;
    Timer upload_timer;
    DMatrix<double> A_d(A_host);
    DVector<double> b_d(b_host);

    Eigen::MatrixXd X0_host(n, cfg.N_CHAINS);
    for (int i = 0; i < cfg.N_CHAINS; ++i) {
        X0_host.col(i) = x0_host;
    }
    DMatrix<double> X_d(X0_host);
    profile.upload_time = upload_timer.elapsed();
    if (verbose) {
        std::cout << "  time            " << std::fixed << std::setprecision(3) << profile.upload_time << "s" << std::endl;
        std::cout << std::endl;
    }

    Eigen::MatrixXd samples_out;

    if (cfg.BACK_TRANSFORM) {
        if (verbose) std::cout << "> loading transform" << std::endl;
        Timer load_transform_timer;
        Eigen::MatrixXd T_host = csv::loadMatrix(cfg.T_FILE);
        Eigen::VectorXd shift_host = csv::loadVector(cfg.SHIFT_FILE);

        int n_orig = T_host.rows();
        profile.original_dim = n_orig;

        if (T_host.cols() != n) {
            throw std::runtime_error("dimension mismatch: transform matrix cols (" + std::to_string(T_host.cols()) + ") != reduced dim (" + std::to_string(n) + ")");
        }

        DMatrix<double> T_d(T_host);
        DVector<double> shift_d(shift_host);
        double load_transform_time = load_transform_timer.elapsed();
        if (verbose) {
            std::cout << "  original_dim    " << n_orig << std::endl;
            std::cout << "  time            " << std::fixed << std::setprecision(3) << load_transform_time << "s" << std::endl;
            std::cout << std::endl;
        }

        if (verbose) std::cout << "> gpu sampling+backmap" << std::endl;
        Timer sampling_timer;

        auto samples_d = CoordinateHitAndRunBackmap(
            A_d, b_d, X_d, T_d, shift_d,
            cfg.N_SAMPLES,
            thinning,
            cfg.N_CHAINS,
            cfg.TPB_SS
        );

        cudaDeviceSynchronize();
        profile.sampling_time = sampling_timer.elapsed();
        profile.backtransform_time = 0.0;
        profile.throughput = (cfg.N_CHAINS * cfg.N_SAMPLES) / profile.sampling_time;

        if (verbose) {
            std::cout << "  time            " << std::fixed << std::setprecision(3) << profile.sampling_time << "s" << std::endl;
            std::cout << "  throughput      " << (int)profile.throughput << " samples/s" << std::endl;
            std::cout << std::endl;
        }

        if (verbose) std::cout << "> downloading" << std::endl;
        Timer download_timer;
        samples_out = samples_d.toHost();
        profile.download_time = download_timer.elapsed();
        if (verbose) {
            std::cout << "  shape           " << samples_out.rows() << " x " << samples_out.cols() << " (original space)" << std::endl;
            std::cout << "  time            " << std::fixed << std::setprecision(3) << profile.download_time << "s" << std::endl;
            std::cout << std::endl;
        }
    } else {
        if (verbose) std::cout << "> gpu sampling" << std::endl;
        Timer sampling_timer;

        auto samples_d = CoordinateHitAndRun(
            A_d, b_d, X_d,
            cfg.N_SAMPLES,
            thinning,
            cfg.N_CHAINS,
            cfg.TPB_SS
        );

        cudaDeviceSynchronize();
        profile.sampling_time = sampling_timer.elapsed();
        profile.backtransform_time = 0.0;
        profile.throughput = (cfg.N_CHAINS * cfg.N_SAMPLES) / profile.sampling_time;

        if (verbose) {
            std::cout << "  time            " << std::fixed << std::setprecision(3) << profile.sampling_time << "s" << std::endl;
            std::cout << "  throughput      " << (int)profile.throughput << " samples/s" << std::endl;
            std::cout << std::endl;
        }

        profile.original_dim = n;

        if (verbose) std::cout << "> downloading" << std::endl;
        Timer download_timer;
        samples_out = samples_d.toHost();
        profile.download_time = download_timer.elapsed();
        if (verbose) {
            std::cout << "  shape           " << samples_out.rows() << " x " << samples_out.cols() << " (reduced space)" << std::endl;
            std::cout << "  time            " << std::fixed << std::setprecision(3) << profile.download_time << "s" << std::endl;
            std::cout << std::endl;
        }
    }

    if (cfg.WRITE_DATA) {
        ensure_dir(cfg.OUT_DIR);

        if (verbose) std::cout << "> writing npy" << std::endl;
        Timer npy_timer;
        try {
            npy::save(cfg.NPY_FILE, samples_out);
        } catch (const std::exception& e) {
            std::cerr << "write failed: " << e.what() << std::endl;
            return 1;
        }
        profile.write_time = npy_timer.elapsed();
        if (verbose) {
            std::cout << "  file            " << cfg.NPY_FILE << std::endl;
            std::cout << "  time            " << std::fixed << std::setprecision(3) << profile.write_time << "s" << std::endl;
            std::cout << std::endl;
        }
    } else {
        profile.write_time = 0.0;
    }

    profile.total_time = total_timer.elapsed();
    profile.write_json(cfg.PROFILE_FILE, cfg);

    std::string snapshot_time = current_timestamp();
    write_config_snapshot(cfg, profile, snapshot_time);

    if (verbose) {
        std::cout << "timing" << std::endl;
        std::cout << "  load            " << std::fixed << std::setprecision(3) << profile.load_time << "s" << std::endl;
        std::cout << "  upload          " << profile.upload_time << "s" << std::endl;
        if (cfg.BACK_TRANSFORM) {
            std::cout << "  sample+backmap  " << profile.sampling_time << "s" << std::endl;
        } else {
            std::cout << "  sample          " << profile.sampling_time << "s" << std::endl;
        }
        std::cout << "  download        " << profile.download_time << "s" << std::endl;
        if (cfg.WRITE_DATA) {
            std::cout << "  write           " << profile.write_time << "s" << std::endl;
        }
        std::cout << "  ____________________________" << std::endl;
        std::cout << "  total           " << profile.total_time << "s" << std::endl;
        std::cout << std::endl;
        std::cout << "output      :: " << make_absolute_path(cfg.OUT_DIR) << std::endl;
    }

    return 0;
}

void bulk_worker(int device_id,
                 const RuntimeConfig& base_cfg,
                 const std::vector<std::string>& jobs,
                 std::atomic<size_t>& next_index,
                 std::vector<JobResult>& results,
                 std::mutex& results_mutex,
                 std::mutex& log_mutex) {
    set_device(device_id);
    while (true) {
        size_t idx = next_index.fetch_add(1);
        if (idx >= jobs.size()) break;
        const std::string& job_name = jobs[idx];
        RuntimeConfig job_cfg = base_cfg;
        job_cfg.MODEL_NAME = job_name;
        job_cfg.GPU_DEVICE = device_id;
        job_cfg.OUT_DIR = make_job_output_dir(base_cfg.OUT_DIR, job_name);
        job_cfg.derive_paths();

        auto start = std::chrono::high_resolution_clock::now();
        JobResult result;
        result.model_name = job_name;
        result.device_id = device_id;
        result.output_dir = job_cfg.OUT_DIR;

        int rc = 0;
        try {
            rc = run_sampling_job(job_cfg, false, false);
            result.success = (rc == 0);
            if (!result.success) {
                result.message = "return code " + std::to_string(rc);
            }
        } catch (const std::exception& e) {
            result.success = false;
            result.message = e.what();
        }
        result.elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();

        {
            std::lock_guard<std::mutex> lock(log_mutex);
            std::cout << "[gpu " << device_id << "] "
                      << job_name << " -> "
                      << (result.success ? "OK" : "FAIL")
                      << " (" << std::fixed << std::setprecision(2) << result.elapsed << "s";
            if (!result.success && !result.message.empty()) {
                std::cout << ", " << result.message;
            }
            std::cout << ")" << std::endl;
        }

        {
            std::lock_guard<std::mutex> lock(results_mutex);
            results.push_back(std::move(result));
        }
    }
}

int run_bulk_mode(RuntimeConfig cfg) {
    ensure_dir(cfg.OUT_DIR);
    std::vector<std::string> jobs = load_bulk_jobs(cfg);
    if (jobs.empty()) {
        std::cerr << "bulk mode requires BULK_MODEL_LIST with at least one model name" << std::endl;
        return 1;
    }
    std::vector<int> gpu_ids = resolve_gpu_ids(cfg);
    std::cout << "bulk sampling :: " << jobs.size() << " jobs across " << gpu_ids.size() << " gpu(s)" << std::endl;
    std::cout << "output root   :: " << make_absolute_path(cfg.OUT_DIR) << std::endl << std::endl;

    std::atomic<size_t> next_index{0};
    std::vector<JobResult> results;
    results.reserve(jobs.size());
    std::mutex results_mutex;
    std::mutex log_mutex;
    std::vector<std::thread> workers;

    for (int device_id : gpu_ids) {
        workers.emplace_back(bulk_worker, device_id, std::cref(cfg), std::cref(jobs),
                             std::ref(next_index), std::ref(results),
                             std::ref(results_mutex), std::ref(log_mutex));
    }

    for (auto& worker : workers) {
        worker.join();
    }

    size_t succeeded = 0;
    for (const auto& r : results) {
        if (r.success) ++succeeded;
    }

    std::cout << "\nsummary" << std::endl;
    std::cout << "  completed : " << succeeded << " / " << jobs.size() << std::endl;
    std::cout << std::endl;

    std::string summary_path = cfg.OUT_DIR + "/bulk_summary.txt";
    std::ofstream summary(summary_path);
    if (summary.is_open()) {
        for (const auto& r : results) {
            summary << r.model_name << ","
                    << (r.success ? "OK" : "FAIL") << ","
                    << std::fixed << std::setprecision(2) << r.elapsed;
            if (!r.message.empty()) {
                summary << "," << r.message;
            }
            summary << "," << r.output_dir << "\n";
        }
        summary.close();
        std::cout << "  wrote summary :: " << summary_path << std::endl;
    } else {
        std::cerr << "WARNING: Could not write bulk summary to " << summary_path << std::endl;
    }

    return succeeded == jobs.size() ? 0 : 2;
}
