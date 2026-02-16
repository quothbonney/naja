#pragma once
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include "utils.h"
#include <vector>
#include <sstream>

struct RuntimeConfig {
    // Parameters that can be loaded from file
    std::string DATA_DIR = "./models";
    std::string MODEL_NAME = "RECON1";
    std::string OUT_DIR = "./out";
    int N_CHAINS = 16;
    int N_SAMPLES = 10000;
    int TPB_SS = 128;
    int GPU_DEVICE = 0;
    bool BACK_TRANSFORM = true;
    bool WRITE_DATA = true;
    bool VERBOSE = true;
    bool STATUS = true;                 // print minimal phase updates (GNU-ish)
    bool BOUNDS_FILTER = false;          // if true, compute validity vs gem/{l_bounds,u_bounds}.csv
    double BOUNDS_EPS = 1e-6;            // tolerance for bounds filtering
    bool WRITE_SAMPLES_VALID = false;    // if true, also write samples_valid.npy
    bool SKIP_EXISTING = false;          // bulk: skip jobs that already have completed outputs
    int THINNING = 0;                   // 0 => auto (n/6), otherwise explicit steps between saved samples
    // Sampler tuning (CHR kernel)
    // PAIR_PROB: probability of taking a 2-sparse "pair direction" step u = (e_i - e_j)/sqrt(2)
    // This is a cheap non-axis move that helps when the feasible set is thin/tilted relative to axes.
    double PAIR_PROB = 0.0;
    // RESYNC_INTERVAL: if >0, recompute slack = b - A*x every RESYNC_INTERVAL iterations (per chain)
    // This is a numerical hygiene knob for long chains.
    int RESYNC_INTERVAL = 0;
    // Iterative rounding-lite (Jacobi 2x2 preconditioner):
    // If ITER_ROUNDING_PASSES > 0, naja runs a short warmup in reduced space, estimates per-pair Jacobi angles,
    // and then mixes in rotated 2-sparse directions using those angles.
    int ITER_ROUNDING_PASSES = 0;
    int ITER_ROUNDING_WARMUP = 0;
    // Relaxation for conditioning-derived extra inequalities:
    // If >0, adds this epsilon to b_extra (i.e. A_extra x <= b_extra + eps) when loading the augmented polytope.
    // This is a prototyping knob to avoid numerical "point polytopes" from over-tightened extra constraints.
    double EXTRA_CONSTRAINT_EPS = 0.0;
    // Global relaxation for ALL inequalities A x <= b:
    // If >0, we replace b with (b + CONSTRAINT_EPS) before sampling.
    // This is a prototyping knob for cases where the feasible set is effectively lower-dimensional
    // (many constraints are equalities) and we want a small "thickness" to recover geometry.
    double CONSTRAINT_EPS = 0.0;
    // Optional precomputed global pair schedule CSV (i,j,c,s). If set, sampling uses it (pair_mode=2).
    std::string PAIR_SCHEDULE;
    // Whether to load *_rounding_extra_{A,b}.csv:
    // - "auto": load if present (default; current behavior)
    // - "ignore": never load (matches cu_sample WT runs)
    // - "require": error if missing
    std::string EXTRA_CONSTRAINTS = "auto";
    // Start policy:
    // - "file": use rounding_start.csv as-is (default).
    // - "cube_center": compute an interior feasible start by solving the max-inscribed axis-aligned cube LP.
    std::string START_POLICY = "file";

    // Affine-hull reduction (experimental):
    // Many rounded polytopes are not full-dimensional in the provided reduced coordinates (A x <= b),
    // which makes hit-and-run along generic directions degenerate (step size collapses).
    //
    // If AFFINE_HULL_TOL > 0, naja detects rows with slack <= tol at the feasible start point and
    // treats them as (approximately) always-tight equalities to compute a nullspace basis B.
    // Sampling then happens in z-coordinates with x = x0 + B z.
    //
    // 0.0 disables this and samples in the provided reduced space as-is.
    double AFFINE_HULL_TOL = 0.0;

    // Sparse-direction escape moves (to overcome axis-misalignment without full rerounding):
    // With probability KSPARSE_PROB, take a k-sparse direction u with random +/- entries of size 1/sqrt(k).
    // This is more robust than pure coordinate HR but still much cheaper than dense random-direction HR.
    double KSPARSE_PROB = 0.0;
    int KSPARSE_K = 8;
    std::string GPU_LIST;
    std::string BULK_MODEL_LIST;
    std::string source_file;

    // Derived paths (populated after loading)
    std::string MODEL_DIR;
    std::string ROUND_PREFIX;
    std::string A_FILE;
    std::string B_FILE;
    std::string START_FILE;
    std::string T_FILE;
    std::string SHIFT_FILE;
    std::string A_EXTRA_FILE;
    std::string B_EXTRA_FILE;
    std::string NPY_FILE;
    std::string PROFILE_FILE;

    void derive_paths() {
        // Remove trailing slash from DATA_DIR if present
        if (!DATA_DIR.empty() && DATA_DIR.back() == '/') {
            DATA_DIR.pop_back();
        }
        if (!OUT_DIR.empty() && OUT_DIR.back() == '/') {
            OUT_DIR.pop_back();
        }

        MODEL_DIR = DATA_DIR + "/" + MODEL_NAME;
        std::string ROUNDING_DIR = MODEL_DIR + "/rounding";
        ROUND_PREFIX = ROUNDING_DIR + "/" + MODEL_NAME + "_rounding";
        
        A_FILE = ROUND_PREFIX + "_A.csv";
        B_FILE = ROUND_PREFIX + "_b.csv";
        START_FILE = ROUND_PREFIX + "_start.csv";
        T_FILE = ROUND_PREFIX + "_T.csv";
        SHIFT_FILE = ROUND_PREFIX + "_shift.csv";
        A_EXTRA_FILE = ROUND_PREFIX + "_extra_A.csv";
        B_EXTRA_FILE = ROUND_PREFIX + "_extra_b.csv";
        
        NPY_FILE = OUT_DIR + "/samples.npy";
        PROFILE_FILE = OUT_DIR + "/profile.json";
    }

    bool is_bulk_mode() const {
        return !BULK_MODEL_LIST.empty();
    }

    static RuntimeConfig load(const std::string& filename) {
        RuntimeConfig cfg;
        cfg.source_file = make_absolute_path(filename);

        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "config not found: " << filename << " (using defaults)" << std::endl;
            cfg.derive_paths();
            return cfg;
        }

        std::string line;
        while (std::getline(file, line)) {
            // Trim whitespace
            size_t start = line.find_first_not_of(" \t");
            if (start == std::string::npos) continue; // Empty line
            
            line = line.substr(start);
            size_t end = line.find_last_not_of(" \t");
            if (end != std::string::npos) {
                line = line.substr(0, end + 1);
            }

            if (line.empty() || line[0] == '#') continue;

            std::istringstream is_line(line);
            std::string key;
            if (std::getline(is_line, key, '=')) {
                std::string value;
                if (std::getline(is_line, value)) {
                    // Trim key
                    size_t k_end = key.find_last_not_of(" \t");
                    if (k_end != std::string::npos) key = key.substr(0, k_end + 1);
                    
                    // Trim value
                    size_t v_start = value.find_first_not_of(" \t");
                    if (v_start != std::string::npos) value = value.substr(v_start);
                    
                    size_t v_end = value.find_last_not_of(" \t");
                    if (v_end != std::string::npos) value = value.substr(0, v_end + 1);
                    
                    // Parse known keys
                    if (key == "DATA_DIR") cfg.DATA_DIR = value;
                    else if (key == "MODEL_NAME") cfg.MODEL_NAME = value;
                    else if (key == "OUT_DIR") cfg.OUT_DIR = value;
                    else if (key == "N_CHAINS") cfg.N_CHAINS = std::stoi(value);
                    else if (key == "N_SAMPLES") cfg.N_SAMPLES = std::stoi(value);
                    else if (key == "TPB_SS") cfg.TPB_SS = std::stoi(value);
                    else if (key == "GPU_DEVICE") cfg.GPU_DEVICE = std::stoi(value);
                    else if (key == "BACK_TRANSFORM") cfg.BACK_TRANSFORM = (value == "true" || value == "1");
                    else if (key == "WRITE_DATA") cfg.WRITE_DATA = (value == "true" || value == "1");
                    else if (key == "VERBOSE") cfg.VERBOSE = !(value == "false" || value == "0");
                    else if (key == "STATUS") cfg.STATUS = !(value == "false" || value == "0");
                    else if (key == "BOUNDS_POLICY") cfg.BOUNDS_FILTER = (value == "filter");
                    else if (key == "BOUNDS_EPS") cfg.BOUNDS_EPS = std::stod(value);
                    else if (key == "WRITE_SAMPLES_VALID") cfg.WRITE_SAMPLES_VALID = (value == "true" || value == "1");
                    else if (key == "SKIP_EXISTING") cfg.SKIP_EXISTING = (value == "true" || value == "1");
                    else if (key == "THINNING") cfg.THINNING = std::stoi(value);
                    else if (key == "PAIR_PROB") cfg.PAIR_PROB = std::stod(value);
                    else if (key == "RESYNC_INTERVAL") cfg.RESYNC_INTERVAL = std::stoi(value);
                    else if (key == "ITER_ROUNDING_PASSES") cfg.ITER_ROUNDING_PASSES = std::stoi(value);
                    else if (key == "ITER_ROUNDING_WARMUP") cfg.ITER_ROUNDING_WARMUP = std::stoi(value);
                    else if (key == "EXTRA_CONSTRAINT_EPS") cfg.EXTRA_CONSTRAINT_EPS = std::stod(value);
                    else if (key == "CONSTRAINT_EPS") cfg.CONSTRAINT_EPS = std::stod(value);
                    else if (key == "PAIR_SCHEDULE") cfg.PAIR_SCHEDULE = value;
                    else if (key == "EXTRA_CONSTRAINTS") cfg.EXTRA_CONSTRAINTS = value;
                    else if (key == "START_POLICY") cfg.START_POLICY = value;
                    else if (key == "AFFINE_HULL_TOL") cfg.AFFINE_HULL_TOL = std::stod(value);
                    else if (key == "KSPARSE_PROB") cfg.KSPARSE_PROB = std::stod(value);
                    else if (key == "KSPARSE_K") cfg.KSPARSE_K = std::stoi(value);
                    else if (key == "GPU_LIST") cfg.GPU_LIST = value;
                    else if (key == "BULK_MODEL_LIST") cfg.BULK_MODEL_LIST = value;
                }
            }
        }
        
        cfg.derive_paths();
        return cfg;
    }
};
