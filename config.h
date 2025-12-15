// Hardcoded configuration for GPU CHR sampling demo
#pragma once

#include <string>

namespace config {
    // Data paths
    // Directory containing outputs from the rounding tool.
    // Adjust DATA_DIR or MODEL_NAME to switch models.
    const std::string DATA_DIR   = "/data/rbg/users/jdcarson/hop/cu_sample/models";
    // Base model name (e.g. "RECON1", "e_coli_core")
    const std::string MODEL_NAME = "e_coli_core";
    // Per-model directory and rounding cache prefix:
    //   MODEL_DIR / (MODEL_NAME + "_rounding_*")
    const std::string MODEL_DIR    = DATA_DIR + std::string("/") + MODEL_NAME;
    const std::string ROUND_PREFIX = MODEL_DIR + std::string("/") + MODEL_NAME + std::string("_rounding");
    const std::string A_FILE       = ROUND_PREFIX + "_A.csv";      // Ay * y <= by
    const std::string B_FILE       = ROUND_PREFIX + "_b.csv";
    const std::string START_FILE   = ROUND_PREFIX + "_start.csv";  // typically zeros
    const std::string T_FILE       = ROUND_PREFIX + "_T.csv";      // backmap: v = T y + shift
    const std::string SHIFT_FILE   = ROUND_PREFIX + "_shift.csv";
    
    // Output
    // Directory for sampler outputs (must be writable and have existing parent dir)
    const std::string OUT_DIR = "/data/rbg/users/jdcarson/hop/cu_sample/out";
    
    // Sampling parameters
    const int N_CHAINS = 16;
    const int N_SAMPLES = 10000;
    const int TPB_SS = 128;
    const int GPU_DEVICE = 1;
    const bool BACK_TRANSFORM = true;
    const bool WRITE_DATA = true;  // Whether to write samples to disk (set false for pure timing)
}
