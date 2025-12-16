#pragma once
#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include "runtime_config.h"

struct JobResult {
    std::string model_name;
    int device_id = 0;
    bool success = false;
    bool skipped = false;
    std::string message;
    double elapsed = 0.0;
    std::string output_dir;
};

// Core function to run a single sampling job (load, sample, write).
// Throws exceptions on failure.
int run_sampling_job(RuntimeConfig cfg, bool verbose, bool show_device_banner);

// Worker function for bulk processing
void bulk_worker(int device_id,
                 const RuntimeConfig& base_cfg,
                 const std::vector<std::string>& jobs,
                 std::atomic<size_t>& next_index,
                 std::vector<JobResult>& results,
                 std::mutex& results_mutex,
                 std::mutex& log_mutex);

// Entry point for bulk execution
int run_bulk_mode(RuntimeConfig cfg);
