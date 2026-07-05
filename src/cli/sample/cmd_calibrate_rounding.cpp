#include "cli/sample/commands.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cli/sample/common.h"
#include "csv_loader.h"
#include "rounding/jacobi.h"
#include "gpu/dmatrix.h"
#include "gpu/dvector.h"
#include "gpu/gpusamplers.h"
#include "pipeline/model_contract.h"
#include "pipeline/model_io.h"
#include "rounding/schedule_io.h"
#include "utils.h"

namespace naja::cli::sample {

void cmd_calibrate_rounding(int argc, char** argv) {
    std::string models_root;
    std::string model_list;
    std::string out_csv;
    int warmup = 2000;
    int passes = 2;
    double pair_prob = 0.3;
    double extra_eps = 0.0;
    int seed = 1;
    int max_models = 0; // 0 => all

    for (int i = 0; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--models-root") models_root = next_arg(i, argc, argv, a);
        else if (a == "--model-list") model_list = next_arg(i, argc, argv, a);
        else if (a == "--out") out_csv = next_arg(i, argc, argv, a);
        else if (a == "--warmup") warmup = std::stoi(next_arg(i, argc, argv, a));
        else if (a == "--passes") passes = std::stoi(next_arg(i, argc, argv, a));
        else if (a == "--pair-prob") pair_prob = std::stod(next_arg(i, argc, argv, a));
        else if (a == "--extra-constraint-eps") extra_eps = std::stod(next_arg(i, argc, argv, a));
        else if (a == "--seed") seed = std::stoi(next_arg(i, argc, argv, a));
        else if (a == "--max-models") max_models = std::stoi(next_arg(i, argc, argv, a));
        else die_usage("unknown flag: " + a);
    }
    if (models_root.empty()) die_usage("missing --models-root");
    if (model_list.empty()) die_usage("missing --model-list");
    if (out_csv.empty()) die_usage("missing --out");
    if (warmup <= 0) die_usage("invalid --warmup");
    if (passes <= 0) die_usage("invalid --passes");
    if (!(pair_prob > 0.0 && pair_prob <= 1.0)) die_usage("invalid --pair-prob (0,1]");
    if (extra_eps < 0.0) die_usage("invalid --extra-constraint-eps (must be >=0)");
    if (max_models < 0) die_usage("invalid --max-models (must be >=0)");
    if (seed == 0) seed = 1;

    auto names = naja::pipeline::load_model_list(model_list);
    if (max_models > 0 && (int)names.size() > max_models) {
        names.resize(max_models);
    }

    // Determine reduced dim from first model.
    const std::string first_dir = make_absolute_path(models_root + "/" + names.front());
    auto first_contract = naja::pipeline::parse_model_dir(first_dir);
    Eigen::MatrixXd A0 = naja::pipeline::RoundingReader(first_contract.rounding_dir, first_contract.model_name).A();
    const int d = (int)A0.cols();
    if (d < 2) {
        throw std::runtime_error("calibrate-rounding requires reduced_dim >= 2");
    }

    // Fixed disjoint pairing (seeded shuffle).
    std::vector<int> perm(d);
    for (int i = 0; i < d; ++i) perm[i] = i;
    std::mt19937_64 rng((uint64_t)seed);
    std::shuffle(perm.begin(), perm.end(), rng);
    const int n_pairs = d / 2;
    naja::rounding::PairSchedule sched;
    sched.i.resize(n_pairs);
    sched.j.resize(n_pairs);
    sched.c.resize(n_pairs);
    sched.s.resize(n_pairs);
    for (int k = 0; k < n_pairs; ++k) {
        sched.i[k] = perm[2 * k + 0];
        sched.j[k] = perm[2 * k + 1];
        sched.c[k] = 1.0;
        sched.s[k] = 0.0;
    }

    for (int pass = 0; pass < passes; ++pass) {
        // Aggregate pair statistics across models (mean of per-sample moments).
        Eigen::VectorXd sum_var_i = Eigen::VectorXd::Zero(n_pairs);
        Eigen::VectorXd sum_var_j = Eigen::VectorXd::Zero(n_pairs);
        Eigen::VectorXd sum_cov = Eigen::VectorXd::Zero(n_pairs);
        Eigen::VectorXd sum_w = Eigen::VectorXd::Zero(n_pairs);

        // Upload current schedule once per pass (reused across models).
        naja::gpu::DVector<int> pair_i_d(sched.i);
        naja::gpu::DVector<int> pair_j_d(sched.j);
        naja::gpu::DVector<double> pair_c_d(sched.c);
        naja::gpu::DVector<double> pair_s_d(sched.s);

        for (size_t mi = 0; mi < names.size(); ++mi) {
            const std::string model_dir = make_absolute_path(models_root + "/" + names[mi]);
            auto c = naja::pipeline::parse_model_dir(model_dir);

            naja::pipeline::RoundingReader reader(c.rounding_dir, c.model_name);
            Eigen::MatrixXd A = reader.A();
            Eigen::VectorXd b = reader.b();
            Eigen::VectorXd x0 = reader.start();

            // Augment with extra constraints if present.
            if (reader.has_extra()) {
                Eigen::MatrixXd A_extra = reader.extra_A();
                Eigen::VectorXd b_extra = reader.extra_b();
                if (extra_eps > 0.0) b_extra.array() += extra_eps;
                Eigen::MatrixXd A_aug(A.rows() + A_extra.rows(), A.cols());
                A_aug.topRows(A.rows()) = A;
                A_aug.bottomRows(A_extra.rows()) = A_extra;
                Eigen::VectorXd b_aug(b.size() + b_extra.size());
                b_aug.head(b.size()) = b;
                b_aug.tail(b_extra.size()) = b_extra;
                A = std::move(A_aug);
                b = std::move(b_aug);
            }

            // Sample warmup in reduced space, using the current schedule as rotated pair directions.
            naja::gpu::DMatrix<double> A_d(A);
            naja::gpu::DVector<double> b_d(b);
            Eigen::MatrixXd X0(d, 1);
            X0.col(0) = x0;
            naja::gpu::DMatrix<double> X_d(X0);

            auto W_d = naja::gpu::CoordinateHitAndRun(
                A_d, b_d, X_d,
                warmup, /*thinning*/ 1, /*nchains*/ 1,
                /*tpb*/ 128,
                seed ^ (int)mi ^ (pass + 1),
                pair_prob,
                /*resync_interval*/ 0,
                /*ksparse_prob*/ 0.0,
                /*ksparse_k*/ 8,
                /*pair_mode*/ 2,
                &pair_i_d, &pair_j_d, &pair_c_d, &pair_s_d
            );
            cudaDeviceSynchronize();
            Eigen::MatrixXd W = W_d.toHost(); // d x warmup

            // Update per-pair covariance moments.
            for (int k = 0; k < n_pairs; ++k) {
                const int ii = sched.i[k];
                const int jj = sched.j[k];
                const auto vi = W.row(ii).array();
                const auto vj = W.row(jj).array();
                const double miu = vi.mean();
                const double mju = vj.mean();
                const auto di = vi - miu;
                const auto dj = vj - mju;
                const double var_i = (di * di).mean();
                const double var_j = (dj * dj).mean();
                const double cov_ij = (di * dj).mean();

                // Weight by 1 (equal weighting of models); could weight by warmup length but fixed here.
                sum_var_i[k] += var_i;
                sum_var_j[k] += var_j;
                sum_cov[k] += cov_ij;
                sum_w[k] += 1.0;
            }
        }

        // Update angles
        for (int k = 0; k < n_pairs; ++k) {
            const double w = sum_w[k];
            if (w <= 0.0) continue;
            const double var_i = sum_var_i[k] / w;
            const double var_j = sum_var_j[k] / w;
            const double cov_ij = sum_cov[k] / w;
            const auto cs = naja::rounding::jacobi_rotation_cs(var_i, var_j, cov_ij);
            sched.c[k] = cs.first;
            sched.s[k] = cs.second;
        }
    }

    naja::rounding::write_pair_schedule_csv(out_csv, sched);
    std::cout << "OK\n";
    std::cout << "out              :: " << make_absolute_path(out_csv) << "\n";
    std::cout << "models           :: " << names.size() << "\n";
    std::cout << "reduced_dim      :: " << d << "\n";
    std::cout << "pairs            :: " << n_pairs << "\n";
    std::cout << "passes           :: " << passes << "\n";
    std::cout << "warmup           :: " << warmup << "\n";
    std::cout << "pair_prob        :: " << pair_prob << "\n";
    std::cout << "extra_eps        :: " << extra_eps << "\n";
}

} // namespace naja::cli::sample


