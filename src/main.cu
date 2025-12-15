// Minimal standalone GPU sampler for RECON1 - sanity check for GPU code changes
#include <iostream>
#include <cstdlib>
#include <vector>

// Local headers
#include "runtime_config.h"
#include "utils.h"
#include "job.h"

int main(int argc, char** argv) {
    RuntimeConfig cfg = RuntimeConfig::load(argc > 1 ? argv[1] : "config.txt");
    const char* env_cfg = std::getenv("NAJA_CONFIG_PATH");
    if (env_cfg && *env_cfg) {
        cfg.source_file = env_cfg;
    }
    const char* env_run_dir = std::getenv("NAJA_RUN_DIR");
    if (env_run_dir && *env_run_dir) {
        cfg.OUT_DIR = env_run_dir;
        cfg.derive_paths();
    }

    ensure_dir(cfg.OUT_DIR);

    if (cfg.is_bulk_mode()) {
        try {
            // Bulk mode logic lives in job.cu now, but we need to expose it.
            // For now, we implemented the worker functions in job.h/cu, 
            // but run_bulk_mode logic was in main.cu and needs to move or be re-implemented.
            
            // Since we moved run_bulk_mode logic to job.cu (impl detail), 
            // let's just call a bulk entry point.
            // Wait, I didn't expose run_bulk_mode in job.h yet.
            // I should add it to job.h
            
            // Re-declaring here to match what I'll put in job.h shortly
            extern int run_bulk_mode(RuntimeConfig cfg);
            return run_bulk_mode(cfg);
        } catch (const std::exception& e) {
            std::cerr << "bulk mode failed: " << e.what() << std::endl;
            return 1;
        }
    }

    return run_sampling_job(cfg, cfg.VERBOSE, true);
}
