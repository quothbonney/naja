#include "job.h"

#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "csv_loader.h"
#include "device_utils.h"
#include "dmatrix.h"
#include "dvector.h"
#include "engine/bounds_filter.h"
#include "gpusamplers.h"
#include "npy.h"
#include "profile.h"
#include "util/status.h"
#include "utils.h"

using namespace naja::gpu;

namespace {

void write_config_snapshot(const RuntimeConfig& cfg, const ProfileData& profile, const std::string& timestamp) {
    const std::string dest = cfg.OUT_DIR + "/config_used.txt";
    std::ifstream src(cfg.source_file);
    std::ofstream dst(dest);
    if (!dst.is_open()) {
        throw std::runtime_error("cannot write config snapshot: " + dest);
    }

    dst << "# Source config: " << cfg.source_file << "\n";
    if (!timestamp.empty()) {
        dst << "# Captured at: " << timestamp << "\n";
    }
    dst << "# Output directory: " << make_absolute_path(cfg.OUT_DIR) << "\n\n";

    if (!src.is_open()) {
        throw std::runtime_error("cannot read source config: " + cfg.source_file);
    }
    dst << src.rdbuf();
    src.close();

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
    dst << "BOUNDS_POLICY=" << (cfg.BOUNDS_FILTER ? "filter" : "ignore") << "\n";
    if (cfg.BOUNDS_FILTER) {
        dst << "BOUNDS_EPS=" << std::setprecision(12) << cfg.BOUNDS_EPS << "\n";
        dst << "WRITE_SAMPLES_VALID=" << (cfg.WRITE_SAMPLES_VALID ? "true" : "false") << "\n";
    }
    dst << std::flush;
}

} // namespace

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

    naja::status::phase(cfg.STATUS, "loading polytope");
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

    naja::status::phase(cfg.STATUS, "uploading to gpu");
    Timer upload_timer;
    DMatrix<double> A_d(A_host);
    DVector<double> b_d(b_host);

    Eigen::MatrixXd X0_host(n, cfg.N_CHAINS);
    for (int i = 0; i < cfg.N_CHAINS; ++i) {
        X0_host.col(i) = x0_host;
    }
    DMatrix<double> X_d(X0_host);
    profile.upload_time = upload_timer.elapsed();

    Eigen::MatrixXd samples_out;

    if (cfg.BACK_TRANSFORM) {
        naja::status::phase(cfg.STATUS, "loading transform");
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
        (void)load_transform_timer;

        naja::status::phase(cfg.STATUS, "gpu sampling+backmap");
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

        naja::status::phase(cfg.STATUS, "downloading");
        Timer download_timer;
        samples_out = samples_d.toHost();
        profile.download_time = download_timer.elapsed();
    } else {
        naja::status::phase(cfg.STATUS, "gpu sampling");
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

        profile.original_dim = n;

        naja::status::phase(cfg.STATUS, "downloading");
        Timer download_timer;
        samples_out = samples_d.toHost();
        profile.download_time = download_timer.elapsed();
    }

    if (cfg.WRITE_DATA) {
        ensure_dir(cfg.OUT_DIR);
        naja::status::phase(cfg.STATUS, "writing npy");
        Timer npy_timer;
        npy::save(cfg.NPY_FILE, samples_out);
        profile.write_time = npy_timer.elapsed();
    } else {
        profile.write_time = 0.0;
    }

    if (cfg.BOUNDS_FILTER) {
        naja::status::phase(cfg.STATUS, "bounds filter (GEM validity)");
        naja::engine::bounds_filter_and_write(cfg, samples_out);
    }

    profile.total_time = total_timer.elapsed();
    profile.write_json(cfg.PROFILE_FILE, cfg);

    std::string snapshot_time = current_timestamp();
    write_config_snapshot(cfg, profile, snapshot_time);

    if (verbose) {
        std::cout << "output      :: " << make_absolute_path(cfg.OUT_DIR) << std::endl;
    }

    return 0;
}
