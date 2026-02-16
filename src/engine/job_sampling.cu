#include "job.h"

#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "csv_loader.h"
#include "device_utils.h"
#include "dmatrix.h"
#include "dvector.h"
#include "engine/bounds_filter.h"
#include "engine/start_feasibility.h"
#include "engine/iterative_rounding.h"
#include "engine/extra_constraints.h"
#include "gpusamplers.h"
#include "npy.h"
#include "profile.h"
#include "util/status.h"
#include "utils.h"
#include "pipeline/pair_schedule.h"
#include "pipeline/feasible_start_lp.h"

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
        if (cfg.PAIR_PROB > 0.0) {
            std::cout << "  pair_prob       " << cfg.PAIR_PROB << std::endl;
        }
        if (cfg.RESYNC_INTERVAL > 0) {
            std::cout << "  resync_interval " << cfg.RESYNC_INTERVAL << std::endl;
        }
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
    bool extra_loaded = false;
    Eigen::MatrixXd A_extra;
    Eigen::VectorXd b_extra;
    Eigen::MatrixXd* A_extra_ptr = nullptr;
    Eigen::VectorXd* b_extra_ptr = nullptr;
    if (has_extra_A || has_extra_b) {
        if (!has_extra_A || !has_extra_b) {
            throw std::runtime_error("incomplete extra constraint files (need both A_extra and b_extra)");
        }
        A_extra = csv::loadMatrix(cfg.A_EXTRA_FILE);
        b_extra = csv::loadVector(cfg.B_EXTRA_FILE);
        A_extra_ptr = &A_extra;
        b_extra_ptr = &b_extra;
    }

    naja::engine::ExtraConstraintsMode extra_mode = naja::engine::parse_extra_constraints_mode(cfg.EXTRA_CONSTRAINTS);
    extra_loaded = naja::engine::maybe_augment_extra_constraints(A_host, b_host, A_extra_ptr, b_extra_ptr, extra_mode, cfg.EXTRA_CONSTRAINT_EPS);
    if (extra_loaded && verbose) {
        std::cout << "  + extra constraints : " << (A_extra_ptr ? A_extra_ptr->rows() : 0) << std::endl;
    }
    if (cfg.CONSTRAINT_EPS > 0.0) {
        // Global relaxation: A x <= b + eps.
        b_host.array() += cfg.CONSTRAINT_EPS;
        if (verbose) {
            std::cout << "  constraint_eps   " << std::setprecision(12) << cfg.CONSTRAINT_EPS << "\n";
        }
    }
    profile.load_time = load_timer.elapsed();

    int m = A_host.rows();
    int n = A_host.cols();
    const int n_reduced_file = n;

    if (x0_host.size() != n) {
        throw std::runtime_error("dimension mismatch: start point size (" + std::to_string(x0_host.size()) + ") != cols (" + std::to_string(n) + ")");
    }
    if (b_host.size() != m) {
        throw std::runtime_error("dimension mismatch: rhs vector size (" + std::to_string(b_host.size()) + ") != rows (" + std::to_string(m) + ")");
    }

    // CHR requires a feasible start point for the constraint set we actually sample.
    // The stored rounding_start.csv is often garbage for A+extra; allow an LP-based interior start.
    if (cfg.START_POLICY == "cube_center") {
        naja::status::phase(cfg.STATUS, "start policy: cube_center");
        auto center = naja::pipeline::axis_aligned_cube_center_lp_gurobi(A_host, b_host);
        x0_host = center.first;
        if (verbose) {
            std::cout << "  start_cube_r     " << std::setprecision(6) << center.second << "\n";
        }
    } else if (cfg.START_POLICY != "file") {
        throw std::runtime_error("invalid START_POLICY: " + cfg.START_POLICY);
    }
    naja::engine::require_feasible_start(A_host, b_host, x0_host, 1e-9, extra_loaded ? "A+extra" : "A");

    // Optional affine-hull reduction: if the feasible set is not full-dimensional in the provided
    // reduced coordinates, coordinate/pair HR will almost surely propose directions outside the
    // affine hull and get a degenerate (near-zero) step size. Detect near-tight constraints at x0,
    // compute a nullspace basis, and sample in that nullspace.
    bool hull_enabled = false;
    Eigen::MatrixXd hull_basis;   // (n_reduced_file x d)
    Eigen::VectorXd hull_shift;   // (n_reduced_file)
    if (cfg.AFFINE_HULL_TOL > 0.0 && n > 0) {
        Eigen::VectorXd slack0 = b_host - (A_host * x0_host);
        std::vector<int> tight_rows;
        tight_rows.reserve((size_t)m);
        for (int i = 0; i < m; ++i) {
            if (slack0[i] <= cfg.AFFINE_HULL_TOL) tight_rows.push_back(i);
        }
        if (!tight_rows.empty()) {
            Eigen::MatrixXd Aeq((int)tight_rows.size(), n);
            for (int r = 0; r < (int)tight_rows.size(); ++r) {
                Aeq.row(r) = A_host.row(tight_rows[r]);
            }
            Eigen::FullPivLU<Eigen::MatrixXd> lu(Aeq);
            // Use a fairly loose threshold; these rows are already selected by slack tolerance and
            // we only need an approximate affine hull basis for sampling.
            lu.setThreshold(1e-10);
            Eigen::MatrixXd B = lu.kernel(); // (n x d)
            const int d = (int)B.cols();
            if (d > 0 && d < n) {
                hull_enabled = true;
                hull_basis = std::move(B);
                hull_shift = x0_host;

                // Transform: y = y0 + B z
                // A(y0 + B z) <= b  =>  (A B) z <= (b - A y0) = slack0
                A_host = A_host * hull_basis;
                b_host = slack0;
                x0_host = Eigen::VectorXd::Zero(d);
                m = A_host.rows();
                n = A_host.cols();

                if (verbose) {
                    std::cout << "  affine_hull_tol  " << std::setprecision(12) << cfg.AFFINE_HULL_TOL << "\n";
                    std::cout << "  affine_hull_rows " << tight_rows.size() << "\n";
                    std::cout << "  affine_hull_dim  " << d << " (from " << n_reduced_file << ")\n";
                }
                naja::engine::require_feasible_start(A_host, b_host, x0_host, 1e-9, "affine-hull reduced");
            } else if (verbose) {
                std::cout << "  affine_hull_tol  " << std::setprecision(12) << cfg.AFFINE_HULL_TOL << " (no reduction; kernel dim=" << d << ")\n";
            }
        } else if (verbose) {
            std::cout << "  affine_hull_tol  " << std::setprecision(12) << cfg.AFFINE_HULL_TOL << " (no tight rows)\n";
        }
    }

    int thinning = (cfg.THINNING > 0) ? cfg.THINNING : std::max(n / 6, 1);
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

    // Seed: ensure jobs do not share identical RNG streams by default.
    // If cfg.SEED is set nonzero, use it; otherwise derive from model+gpu.
    std::hash<std::string> h;
    int seed = (int)(h(cfg.MODEL_NAME) ^ (h(cfg.MODEL_DIR) << 1) ^ (uint64_t)(cfg.GPU_DEVICE * 0x9e3779b9));
    if (seed == 0) seed = 1;

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
    // Optional iterative rounding-lite: compute Jacobi pair angles from a short warmup in reduced space.
    // This does not fix near lower-dimensional polytopes; it targets axis-misalignment slow mixing.
    std::unique_ptr<naja::gpu::DVector<int>> pair_i_d;
    std::unique_ptr<naja::gpu::DVector<int>> pair_j_d;
    std::unique_ptr<naja::gpu::DVector<double>> pair_c_d;
    std::unique_ptr<naja::gpu::DVector<double>> pair_s_d;
    int pair_mode = 1; // 1=fixed (ei-ej)/sqrt(2); 2=jacobi rotated pairs

    // If a precomputed schedule is provided, use it and skip per-model iterative rounding.
    if (!cfg.PAIR_SCHEDULE.empty()) {
        naja::status::phase(cfg.STATUS, "loading pair schedule");
        auto sched = naja::pipeline::load_pair_schedule_csv(cfg.PAIR_SCHEDULE);
        pair_i_d = std::make_unique<naja::gpu::DVector<int>>(sched.i);
        pair_j_d = std::make_unique<naja::gpu::DVector<int>>(sched.j);
        pair_c_d = std::make_unique<naja::gpu::DVector<double>>(sched.c);
        pair_s_d = std::make_unique<naja::gpu::DVector<double>>(sched.s);
        pair_mode = 2;
    }

    if (cfg.PAIR_SCHEDULE.empty() && cfg.ITER_ROUNDING_PASSES > 0 && cfg.ITER_ROUNDING_WARMUP > 0 && n >= 2) {
        naja::status::phase(cfg.STATUS, "iterative rounding-lite (warmup)");
        // Build disjoint pairs once (fixed pairing) and update angles per pass.
        std::vector<int> perm(n);
        for (int i = 0; i < n; ++i) perm[i] = i;
        std::mt19937_64 rng(static_cast<uint64_t>(seed) ^ 0x9e3779b97f4a7c15ULL);
        std::shuffle(perm.begin(), perm.end(), rng);
        const int n_pairs = n / 2;
        Eigen::VectorXi pair_i_h(n_pairs), pair_j_h(n_pairs);
        for (int k = 0; k < n_pairs; ++k) {
            pair_i_h[k] = perm[2 * k + 0];
            pair_j_h[k] = perm[2 * k + 1];
        }
        Eigen::VectorXd pair_c_h(n_pairs), pair_s_h(n_pairs);
        pair_c_h.setOnes();
        pair_s_h.setZero();

        // Warmup uses 1 chain for stability/cheapness.
        const int warmup_chains = 1;
        Eigen::MatrixXd X0_warmup(n, warmup_chains);
        X0_warmup.col(0) = x0_host;
        naja::gpu::DMatrix<double> X_warmup_d(X0_warmup);

        // Iterate: sample -> estimate per-pair covariance -> update angles.
        for (int pass = 0; pass < cfg.ITER_ROUNDING_PASSES; ++pass) {
            // Use fixed pair moves in the very first pass to avoid getting completely stuck.
            const int warmup_pair_mode = (pass == 0) ? 1 : 2;
            const double warmup_pair_prob = std::min(std::max(cfg.PAIR_PROB, 0.05), 0.5);

            std::unique_ptr<naja::gpu::DVector<int>> pi_tmp;
            std::unique_ptr<naja::gpu::DVector<int>> pj_tmp;
            std::unique_ptr<naja::gpu::DVector<double>> pc_tmp;
            std::unique_ptr<naja::gpu::DVector<double>> ps_tmp;
            if (warmup_pair_mode == 2) {
                pi_tmp = std::make_unique<naja::gpu::DVector<int>>(pair_i_h);
                pj_tmp = std::make_unique<naja::gpu::DVector<int>>(pair_j_h);
                pc_tmp = std::make_unique<naja::gpu::DVector<double>>(pair_c_h);
                ps_tmp = std::make_unique<naja::gpu::DVector<double>>(pair_s_h);
            }

            auto warmup_samples_d = CoordinateHitAndRun(
                A_d, b_d, X_warmup_d,
                cfg.ITER_ROUNDING_WARMUP,
                /*thinning*/ 1,
                warmup_chains,
                cfg.TPB_SS,
                seed ^ (pass + 1),
                warmup_pair_prob,
                /*resync_interval*/ 0,
                /*ksparse_prob*/ 0.0,
                /*ksparse_k*/ 8,
                warmup_pair_mode,
                pi_tmp ? pi_tmp.get() : nullptr,
                pj_tmp ? pj_tmp.get() : nullptr,
                pc_tmp ? pc_tmp.get() : nullptr,
                ps_tmp ? ps_tmp.get() : nullptr
            );
            cudaDeviceSynchronize();
            Eigen::MatrixXd W = warmup_samples_d.toHost(); // n x warmup_samples

            // Estimate angles per pair.
            for (int k = 0; k < n_pairs; ++k) {
                const int ii = pair_i_h[k];
                const int jj = pair_j_h[k];
                const auto vi = W.row(ii).array();
                const auto vj = W.row(jj).array();
                const double mi = vi.mean();
                const double mj = vj.mean();
                const auto di = vi - mi;
                const auto dj = vj - mj;
                const double var_i = (di * di).mean();
                const double var_j = (dj * dj).mean();
                const double cov_ij = (di * dj).mean();
                const auto cs = naja::engine::jacobi_rotation_cs(var_i, var_j, cov_ij);
                pair_c_h[k] = cs.first;
                pair_s_h[k] = cs.second;
            }
        }

        pair_i_d = std::make_unique<naja::gpu::DVector<int>>(pair_i_h);
        pair_j_d = std::make_unique<naja::gpu::DVector<int>>(pair_j_h);
        pair_c_d = std::make_unique<naja::gpu::DVector<double>>(pair_c_h);
        pair_s_d = std::make_unique<naja::gpu::DVector<double>>(pair_s_h);
        pair_mode = 2;
        if (verbose) {
            std::cout << "  iter_rounding_passes " << cfg.ITER_ROUNDING_PASSES << "\n";
            std::cout << "  iter_rounding_warmup " << cfg.ITER_ROUNDING_WARMUP << "\n";
            std::cout << "  iter_rounding_pairs  " << (n / 2) << "\n";
        }
    }

    if (cfg.BACK_TRANSFORM) {
        naja::status::phase(cfg.STATUS, "loading transform");
        Timer load_transform_timer;
        Eigen::MatrixXd T_host = csv::loadMatrix(cfg.T_FILE);
        Eigen::VectorXd shift_host = csv::loadVector(cfg.SHIFT_FILE);

        int n_orig = T_host.rows();
        profile.original_dim = n_orig;

        const int expected_cols = hull_enabled ? n_reduced_file : n;
        if (T_host.cols() != expected_cols) {
            throw std::runtime_error("dimension mismatch: transform matrix cols (" + std::to_string(T_host.cols()) + ") != reduced dim (" + std::to_string(expected_cols) + ")");
        }
        if (hull_enabled) {
            // Compose: x = T*(y0 + B z) + shift = (T B) z + (T y0 + shift)
            Eigen::MatrixXd T0 = T_host;
            T_host = T0 * hull_basis;
            shift_host = (T0 * hull_shift) + shift_host;
        }
        // After optional composition, transform cols must match sampling dim.
        if (T_host.cols() != n) {
            throw std::runtime_error("internal error: composed transform cols (" + std::to_string(T_host.cols()) + ") != sampling dim (" + std::to_string(n) + ")");
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
            cfg.TPB_SS,
            seed,
            cfg.PAIR_PROB,
            cfg.RESYNC_INTERVAL,
            cfg.KSPARSE_PROB,
            cfg.KSPARSE_K,
            pair_mode,
            pair_i_d ? pair_i_d.get() : nullptr,
            pair_j_d ? pair_j_d.get() : nullptr,
            pair_c_d ? pair_c_d.get() : nullptr,
            pair_s_d ? pair_s_d.get() : nullptr
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
        // If affine-hull reduction is enabled, we still want to map z -> original reduced y (x = y0 + B z),
        // so downstream tools see the expected reduced coordinates rather than the hull coordinates.
        if (hull_enabled) {
            naja::status::phase(cfg.STATUS, "gpu sampling+backmap (affine hull)");
            Timer sampling_timer;
            Eigen::MatrixXd T_host = hull_basis;
            Eigen::VectorXd shift_host = hull_shift;
            DMatrix<double> T_d(T_host);
            DVector<double> shift_d(shift_host);
            auto samples_d = CoordinateHitAndRunBackmap(
                A_d, b_d, X_d, T_d, shift_d,
                cfg.N_SAMPLES,
                thinning,
                cfg.N_CHAINS,
                cfg.TPB_SS,
                seed,
                cfg.PAIR_PROB,
                cfg.RESYNC_INTERVAL,
                cfg.KSPARSE_PROB,
                cfg.KSPARSE_K,
                pair_mode,
                pair_i_d ? pair_i_d.get() : nullptr,
                pair_j_d ? pair_j_d.get() : nullptr,
                pair_c_d ? pair_c_d.get() : nullptr,
                pair_s_d ? pair_s_d.get() : nullptr
            );
            cudaDeviceSynchronize();
            profile.sampling_time = sampling_timer.elapsed();
            profile.backtransform_time = 0.0;
            profile.throughput = (cfg.N_CHAINS * cfg.N_SAMPLES) / profile.sampling_time;
            profile.original_dim = n_reduced_file;
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
            cfg.TPB_SS,
                seed,
                cfg.PAIR_PROB,
                cfg.RESYNC_INTERVAL,
                cfg.KSPARSE_PROB,
                cfg.KSPARSE_K,
                pair_mode,
                pair_i_d ? pair_i_d.get() : nullptr,
                pair_j_d ? pair_j_d.get() : nullptr,
                pair_c_d ? pair_c_d.get() : nullptr,
                pair_s_d ? pair_s_d.get() : nullptr
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
