// Minimal standalone GPU sampler for RECON1 - sanity check for GPU code changes
#include <iostream>
#include <cstdlib>
#include <vector>

// Local headers
#include "runtime_config.h"
#include "utils.h"
#include "job.h"
#include "cli/sample_cli.h"
#include "cli/condition_cli.h"

int main(int argc, char** argv) {
    // New CLI entrypoint: `naja sample ...`
    if (argc >= 2 && std::string(argv[1]) == "sample") {
        // Fail loudly: let unexpected exceptions terminate with a stack trace in the caller.
        return naja_sample_cli_main(argc - 2, argv + 2);
    }
    if (argc >= 2 && std::string(argv[1]) == "condition") {
        return naja_condition_cli_main(argc - 2, argv + 2);
    }

    // Legacy config-driven entrypoint (kept for now):
    //   naja [config_path]
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
        return run_bulk_mode(cfg);
    }
    return run_sampling_job(cfg, cfg.VERBOSE, true);
}
