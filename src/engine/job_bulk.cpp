#include "engine/job.h"

#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <filesystem>
#include <sys/stat.h>

#include "device_utils.h"
#include "util/status.h"
#include "utils.h"

namespace {

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
            ids.push_back(std::stoi(token));
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

bool file_nonempty(const std::string& path) {
    if (!path_exists(path)) return false;
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return false;
    return st.st_size > 0;
}

std::string find_completed_job_dir(const RuntimeConfig& cfg, const std::string& model_name) {
    namespace fs = std::filesystem;
    const std::string model_root = cfg.OUT_DIR + "/" + model_name;
    if (!is_directory(model_root)) return {};

    std::string best;
    std::time_t best_mtime = 0;
    for (const auto& ent : fs::directory_iterator(model_root)) {
        if (!ent.is_directory()) continue;
        const std::string dir = ent.path().string();
        if (!file_nonempty(dir + "/samples.npy")) continue;
        if (!file_nonempty(dir + "/profile.json")) continue;
        if (cfg.BOUNDS_FILTER) {
            if (!file_nonempty(dir + "/valid_mask.npy")) continue;
            if (!file_nonempty(dir + "/valid_fraction.txt")) continue;
            if (!file_nonempty(dir + "/bounds_report.json")) continue;
        }
        struct stat st;
        if (stat(dir.c_str(), &st) != 0) continue;
        if (best.empty() || st.st_mtime > best_mtime) {
            best = dir;
            best_mtime = st.st_mtime;
        }
    }
    return best;
}

} // namespace

void bulk_worker(int device_id,
                 const RuntimeConfig& base_cfg,
                 const std::vector<std::string>& jobs,
                 std::atomic<size_t>& next_index,
                 std::vector<JobResult>& results,
                 std::mutex& results_mutex,
                 std::mutex& log_mutex) {
    naja::gpu::set_device(device_id);
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

        if (base_cfg.SKIP_EXISTING) {
            std::string done = find_completed_job_dir(base_cfg, job_name);
            if (!done.empty()) {
                result.success = true;
                result.skipped = true;
                result.message = "SKIP";
                result.elapsed = 0.0;
                result.output_dir = done;
                {
                    std::lock_guard<std::mutex> lock(log_mutex);
                    std::cout << "[gpu " << device_id << "] "
                              << job_name << " -> SKIP" << std::endl;
                }
                {
                    std::lock_guard<std::mutex> lock(results_mutex);
                    results.push_back(std::move(result));
                }
                continue;
            }
        }

        try {
            int rc = run_sampling_job(job_cfg, false, false);
            result.success = (rc == 0);
            if (!result.success) {
                result.message = "return code " + std::to_string(rc);
            }
        } catch (const std::runtime_error& e) {
            result.success = false;
            result.message = e.what();
        }
        result.elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();

        {
            std::lock_guard<std::mutex> lock(log_mutex);
            std::cout << "[gpu " << device_id << "] "
                      << job_name << " -> "
                      << (result.skipped ? "SKIP" : (result.success ? "OK" : "FAIL"))
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

    std::string summary_path = cfg.OUT_DIR + "/bulk_summary.csv";
    std::ofstream summary(summary_path);
    if (summary.is_open()) {
        summary << "model,status,elapsed_s,device_id,message,output_dir,inherited_from_base\n";
        for (const auto& r : results) {
            std::string inherited_from_base;
            {
                std::string p = cfg.DATA_DIR + "/" + r.model_name + "/rounding/INHERITED_FROM.txt";
                if (path_exists(p)) {
                    std::ifstream f(p);
                    std::string line;
                    while (std::getline(f, line)) {
                        line = trim_copy(line);
                        if (line.rfind("base_model_dir=", 0) == 0) {
                            inherited_from_base = line.substr(std::string("base_model_dir=").size());
                            break;
                        }
                    }
                }
            }
            summary << r.model_name << ","
                    << (r.skipped ? "SKIP" : (r.success ? "OK" : "FAIL")) << ","
                    << std::fixed << std::setprecision(2) << r.elapsed << ","
                    << r.device_id;
            if (!r.message.empty()) {
                summary << "," << r.message;
            } else {
                summary << ",";
            }
            summary << "," << r.output_dir << ","
                    << inherited_from_base << "\n";
        }
        summary.close();
        std::cout << "  wrote summary :: " << summary_path << std::endl;
    } else {
        throw std::runtime_error("cannot write bulk summary: " + summary_path);
    }

    return succeeded == jobs.size() ? 0 : 2;
}


