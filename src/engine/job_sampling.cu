#include "engine/job.h"

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
#include "pipeline/extra_constraints.h"
#include "util/start_feasibility.h"
#include "gpusamplers.h"
#include "npy.h"
#include "engine/profile.h"
#include "rounding/barrier_rotation.h"
#include "rounding/barrier_schedule.h"
#include "rounding/config.h"
#include "rounding/dikin_directions.h"
#include "rounding/plan.h"
#include "util/status.h"
#include "utils.h"
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

    naja::pipeline::ExtraConstraintsMode extra_mode = naja::pipeline::parse_extra_constraints_mode(cfg.EXTRA_CONSTRAINTS);
    extra_loaded = naja::pipeline::maybe_augment_extra_constraints(A_host, b_host, A_extra_ptr, b_extra_ptr, extra_mode, cfg.EXTRA_CONSTRAINT_EPS);
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
    naja::util::require_feasible_start(A_host, b_host, x0_host, 1e-9, extra_loaded ? "A+extra" : "A");

    // Degenerate-rounding check: if tight constraints span the full dimension,
    // the polytope is a point and sampling will produce identical copies.
    // This is a warning (not an abort) so bulk jobs don't die from one bad model.
    {
        const double tight_tol = 1e-6;
        Eigen::VectorXd slack_check = b_host - (A_host * x0_host);
        int n_tight = 0;
        for (int i = 0; i < m; ++i) {
            if (slack_check[i] < tight_tol) ++n_tight;
        }
        if (n_tight >= n) {
            std::cerr << "\n*** WARNING: polytope appears degenerate for model " << cfg.MODEL_NAME << " ***\n"
                      << "  tight constraints: " << n_tight << " / " << m << "\n"
                      << "  reduced dimension: " << n << "\n"
                      << "  The feasible set may be a single point. Samples will likely be identical.\n"
                      << "  Check rounding quality (tight constraint rank vs dimension).\n\n";
        }
    }

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
            lu.setThreshold(1e-10);
            Eigen::MatrixXd B = lu.kernel(); // (n x d)
            const int d = (int)B.cols();
            if (d > 0 && d < n) {
                hull_enabled = true;
                hull_basis = std::move(B);
                hull_shift = x0_host;

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
                naja::util::require_feasible_start(A_host, b_host, x0_host, 1e-9, "affine-hull reduced");
            } else if (verbose) {
                std::cout << "  affine_hull_tol  " << std::setprecision(12) << cfg.AFFINE_HULL_TOL << " (no reduction; kernel dim=" << d << ")\n";
            }
        } else if (verbose) {
            std::cout << "  affine_hull_tol  " << std::setprecision(12) << cfg.AFFINE_HULL_TOL << " (no tight rows)\n";
        }
    }

    // --- Barrier rotation: rotate to eigenbasis of H(x_c) = A'diag(1/s^2)A ---
    // Applied AFTER affine-hull reduction so eigenvectors are in the final reduced space.
    // Pure orthogonal Q: chord lengths are preserved, no direction is broken.
    //
    // Uses ALL constraint rows (base + extra) with current b (eps-inflated if
    // --constraint-eps is set).  The eps helps by evening out Hessian weights
    // so no single tight constraint dominates.  Computing Q from the full
    // augmented polytope outperforms base-only Q because it captures KO-induced
    // coupling that would otherwise be ignored.
    bool barrier_rotated = false;
    bool barrier_whitened = false;
    Eigen::MatrixXd rotation_Q;
    Eigen::VectorXd whiten_scale;       // S_i = 1/sqrt(clamp(lambda_i))
    double barrier_condition = 1.0;     // pre-whitening condition number
    if (cfg.BARRIER_ROTATE && n >= 2) {
        naja::status::phase(cfg.STATUS, "barrier rotation (eigenbasis of H)");
        auto br = naja::rounding::compute_barrier_rotation(A_host, b_host, x0_host, verbose);
        rotation_Q = br.Q;
        barrier_condition = br.eigenvalues[n - 1] / std::max(br.eigenvalues[0], 1e-30);

        // 1) Pure orthogonal rotation into the eigenbasis.
        A_host = A_host * rotation_Q;
        x0_host = rotation_Q.transpose() * x0_host;
        if (hull_enabled) {
            hull_basis = hull_basis * rotation_Q;
        }
        barrier_rotated = true;
        naja::util::require_feasible_start(A_host, b_host, x0_host, 1e-9, "barrier-rotated");

        // 2) Optional diagonal whitening: scale dim i by 1/sqrt(lambda_i).
        //    After this, the local barrier metric at x_c is ~ I.
        //    Clamp eigenvalues so tiny values don't create extreme scales.
        if (cfg.BARRIER_WHITEN) {
            naja::status::phase(cfg.STATUS, "barrier whitening (diagonal rescale)");
            double ev_median = br.eigenvalues[n / 2];
            double ev_floor = std::max(ev_median * 1e-6, 1e-30);

            whiten_scale.resize(n);
            int n_clamped_ev = 0;
            for (int i = 0; i < n; ++i) {
                double ev = br.eigenvalues[i];
                if (ev < ev_floor) { ev = ev_floor; ++n_clamped_ev; }
                whiten_scale[i] = 1.0 / std::sqrt(ev);
            }

            // A_new = A_rot * diag(S),  x_new = diag(1/S) * x_rot
            A_host = A_host * whiten_scale.asDiagonal();
            x0_host = x0_host.cwiseQuotient(whiten_scale);
            if (hull_enabled) {
                hull_basis = hull_basis * whiten_scale.asDiagonal();
            }
            barrier_whitened = true;
            naja::util::require_feasible_start(A_host, b_host, x0_host, 1e-9, "barrier-whitened");

            if (verbose) {
                double post_cond = whiten_scale.maxCoeff() / std::max(whiten_scale.minCoeff(), 1e-30);
                std::cout << "  barrier_whiten: eigenvalue_clamp=" << n_clamped_ev
                          << " scale_range=[" << std::scientific << std::setprecision(3)
                          << whiten_scale.minCoeff() << ", " << whiten_scale.maxCoeff()
                          << "] post_cond=" << std::fixed << std::setprecision(1) << post_cond * post_cond
                          << "\n";
            }
        }
    }

    // --- Dikin escape directions (computed in the final sampling space) ---
    naja::rounding::DikinDirections dikin_dirs;
    int n_base_constraints = 0;
    if (cfg.PAIR_SCHEDULE_METHOD == "barrier" && n >= 2) {
        naja::status::phase(cfg.STATUS, "barrier dikin directions (full Hessian)");
        dikin_dirs = naja::rounding::compute_dikin_directions(A_host, b_host, x0_host, 0, 32, 1e15, 1.0);
        if (dikin_dirs.k > 0 && verbose) {
            std::cout << "  dikin_directions " << dikin_dirs.k << " escape directions (full Hessian)\n";
        }
    } else if (extra_loaded && n >= 2) {
        naja::status::phase(cfg.STATUS, "dikin directions (delta)");
        n_base_constraints = m - (A_extra_ptr ? (int)A_extra_ptr->rows() : 0);
        dikin_dirs = naja::rounding::compute_dikin_directions(A_host, b_host, x0_host, n_base_constraints);
        if (dikin_dirs.k > 0 && verbose) {
            std::cout << "  dikin_directions " << dikin_dirs.k << " escape directions (delta)\n";
        }
    }

    int thinning;
    if (cfg.THINNING > 0) {
        thinning = cfg.THINNING;
    } else if (cfg.BARRIER_ROTATE && barrier_condition > 1.0) {
        // Adaptive thinning: the barrier condition number tells us how
        // anisotropic the geometry is.  Whitening fixes the local metric
        // but global anisotropy still requires more decorrelation steps.
        if (barrier_condition <= 1e4)       thinning = std::max(n / 6, 50);
        else if (barrier_condition <= 1e8)  thinning = 200;
        else                                thinning = 500;
    } else {
        thinning = std::max(n / 6, 1);
    }
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

    // Upload Dikin directions to GPU if available
    std::unique_ptr<naja::gpu::DMatrix<double>> dikin_Av_d;
    std::unique_ptr<naja::gpu::DMatrix<double>> dikin_V_d;
    double dikin_prob = 0.0;
    if (dikin_dirs.k > 0) {
        dikin_Av_d = std::make_unique<naja::gpu::DMatrix<double>>(dikin_dirs.Av);
        dikin_V_d = std::make_unique<naja::gpu::DMatrix<double>>(dikin_dirs.V);
        dikin_prob = 0.15;  // 15% of steps use Dikin escape directions
    }

    profile.upload_time = upload_timer.elapsed();

    Eigen::MatrixXd samples_out;
    naja::rounding::RoundingPlan rounding_plan;

    if (cfg.PAIR_SCHEDULE_METHOD == "barrier" && n >= 2) {
        naja::status::phase(cfg.STATUS, "barrier-Hessian Jacobi schedule");
        auto sched = naja::rounding::compute_barrier_pair_schedule(A_host, b_host, x0_host, verbose);
        rounding_plan.pair_i_d = std::make_unique<naja::gpu::DVector<int>>(sched.i);
        rounding_plan.pair_j_d = std::make_unique<naja::gpu::DVector<int>>(sched.j);
        rounding_plan.pair_c_d = std::make_unique<naja::gpu::DVector<double>>(sched.c);
        rounding_plan.pair_s_d = std::make_unique<naja::gpu::DVector<double>>(sched.s);
        rounding_plan.pair_mode = 2;
        // Auto-enable pair moves if user didn't set --pair-prob explicitly
        if (cfg.PAIR_PROB <= 0.0) {
            cfg.PAIR_PROB = 0.5;
            if (verbose) std::cout << "  pair_prob (auto)  0.5\n";
        }
    } else {
        naja::rounding::RoundingConfig rounding_cfg = naja::rounding::from_runtime_config(cfg);
        rounding_plan = naja::rounding::build_rounding_plan(
            rounding_cfg, A_d, b_d, x0_host, n, cfg.TPB_SS, seed, cfg.STATUS, verbose);
    }
    int pair_mode = rounding_plan.pair_mode; // 1=fixed (ei-ej)/sqrt(2); 2=jacobi rotated pairs

    // Pre-sampling estimate so the user knows what to expect during the silent GPU phase
    {
        long long total_steps = (long long)cfg.N_CHAINS * (long long)cfg.N_SAMPLES * (long long)thinning;
        if (cfg.STATUS) {
            std::cout << "> sampling " << cfg.N_CHAINS << " chains x "
                      << cfg.N_SAMPLES << " samples (thin=" << thinning
                      << ", " << total_steps << " MCMC steps, "
                      << m << " constraints x " << n << " dims)"
                      << std::endl;
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
            Eigen::MatrixXd T0 = T_host;
            T_host = T0 * hull_basis;  // hull_basis already includes rotation_Q if barrier_rotated
            shift_host = (T0 * hull_shift) + shift_host;
        }
        if (barrier_rotated && !hull_enabled) {
            T_host = T_host * rotation_Q;
            if (barrier_whitened) {
                T_host = T_host * whiten_scale.asDiagonal();
            }
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
            rounding_plan.pair_i_d ? rounding_plan.pair_i_d.get() : nullptr,
            rounding_plan.pair_j_d ? rounding_plan.pair_j_d.get() : nullptr,
            rounding_plan.pair_c_d ? rounding_plan.pair_c_d.get() : nullptr,
            rounding_plan.pair_s_d ? rounding_plan.pair_s_d.get() : nullptr
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
                rounding_plan.pair_i_d ? rounding_plan.pair_i_d.get() : nullptr,
                rounding_plan.pair_j_d ? rounding_plan.pair_j_d.get() : nullptr,
                rounding_plan.pair_c_d ? rounding_plan.pair_c_d.get() : nullptr,
                rounding_plan.pair_s_d ? rounding_plan.pair_s_d.get() : nullptr
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
                rounding_plan.pair_i_d ? rounding_plan.pair_i_d.get() : nullptr,
                rounding_plan.pair_j_d ? rounding_plan.pair_j_d.get() : nullptr,
                rounding_plan.pair_c_d ? rounding_plan.pair_c_d.get() : nullptr,
                rounding_plan.pair_s_d ? rounding_plan.pair_s_d.get() : nullptr,
                dikin_prob,
                dikin_Av_d.get(),
                dikin_V_d.get(),
                dikin_dirs.k
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

        // Undo whitening+rotation so samples are in the original reduced space.
        if (barrier_whitened) {
            samples_out = whiten_scale.asDiagonal() * samples_out;
        }
        if (barrier_rotated) {
            samples_out = rotation_Q * samples_out;
        }
        }
    }

    // Post-sampling summary
    if (cfg.STATUS) {
        std::cout << "> done: " << std::fixed << std::setprecision(1) << profile.sampling_time << "s, "
                  << std::setprecision(0) << profile.throughput << " samples/s" << std::endl;
    }

    if (cfg.WRITE_DATA) {
        ensure_dir(cfg.OUT_DIR);
        naja::status::phase(cfg.STATUS, "writing npy");
        Timer npy_timer;
        // Write as (n_samples, dim) float32 — half the size, contiguous sample rows.
        npy::save_f32_samples(cfg.NPY_FILE, samples_out);
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
