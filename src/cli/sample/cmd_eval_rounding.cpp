#include "cli/sample/commands.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "cli/sample/common.h"
#include "csv_loader.h"
#include "rounding/diagnostics.h"
#include "util/constraint_utils.h"
#include "pipeline/model_contract.h"
#include "utils.h"

namespace naja::cli::sample {

namespace {

static double quantile(Eigen::VectorXd v, double q) {
    std::vector<double> a;
    a.reserve((size_t)v.size());
    for (int i = 0; i < v.size(); ++i) a.push_back(v[i]);
    std::sort(a.begin(), a.end());
    if (a.empty()) return 0.0;
    const double pos = q * (a.size() - 1);
    const size_t lo = (size_t)std::floor(pos);
    const size_t hi = (size_t)std::ceil(pos);
    const double t = pos - (double)lo;
    return a[lo] * (1.0 - t) + a[hi] * t;
}

} // namespace

void cmd_eval_rounding(int argc, char** argv) {
    std::string models_root;
    std::string model_list;
    double extra_eps = 0.0;
    double tight_tol = 1e-9;

    for (int i = 0; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--models-root") models_root = next_arg(i, argc, argv, a);
        else if (a == "--model-list") model_list = next_arg(i, argc, argv, a);
        else if (a == "--extra-constraint-eps") extra_eps = std::stod(next_arg(i, argc, argv, a));
        else if (a == "--tight-tol") tight_tol = std::stod(next_arg(i, argc, argv, a));
        else die_usage("unknown flag: " + a);
    }
    if (models_root.empty()) die_usage("missing --models-root");
    if (model_list.empty()) die_usage("missing --model-list");
    if (extra_eps < 0.0) die_usage("invalid --extra-constraint-eps (must be >=0)");
    if (tight_tol < 0.0) die_usage("invalid --tight-tol (must be >=0)");

    auto names = naja::pipeline::load_model_list(model_list);

    std::cout << "model,reduced_dim,constraints,extra_rows,tight_rank,axis_chord_p10,axis_chord_p50,axis_chord_p90,axis_chord_max\n";

    for (const auto& name : names) {
        const std::string model_dir = make_absolute_path(models_root + "/" + name);
        naja::pipeline::ModelContract c = naja::pipeline::parse_model_dir(model_dir);

        const std::string A_path = c.rounding_dir + "/" + c.model_name + "_rounding_A.csv";
        const std::string b_path = c.rounding_dir + "/" + c.model_name + "_rounding_b.csv";
        const std::string start_path = c.rounding_dir + "/" + c.model_name + "_rounding_start.csv";
        const std::string extra_A_path = c.rounding_dir + "/" + c.model_name + "_rounding_extra_A.csv";
        const std::string extra_b_path = c.rounding_dir + "/" + c.model_name + "_rounding_extra_b.csv";

        Eigen::MatrixXd A = csv::loadMatrix(A_path);
        Eigen::VectorXd b = csv::loadVector(b_path);
        Eigen::VectorXd x0 = csv::loadVector(start_path);

        int extra_rows = 0;
        if (path_exists(extra_A_path) && path_exists(extra_b_path)) {
            Eigen::MatrixXd A_extra = csv::loadMatrix(extra_A_path);
            Eigen::VectorXd b_extra = csv::loadVector(extra_b_path);
            if (extra_eps > 0.0) b_extra.array() += extra_eps;
            extra_rows = (int)b_extra.size();

            Eigen::MatrixXd A_aug(A.rows() + A_extra.rows(), A.cols());
            A_aug.topRows(A.rows()) = A;
            A_aug.bottomRows(A_extra.rows()) = A_extra;
            Eigen::VectorXd b_aug(b.size() + b_extra.size());
            b_aug.head(b.size()) = b;
            b_aug.tail(b_extra.size()) = b_extra;
            A = std::move(A_aug);
            b = std::move(b_aug);
        }

        const int tight_rank = naja::util::tight_constraint_rank(A, b, x0, tight_tol);
        const auto chords = naja::rounding::chr_axis_chords(A, b, x0).chord_len;
        const double p10 = quantile(chords, 0.10);
        const double p50 = quantile(chords, 0.50);
        const double p90 = quantile(chords, 0.90);
        const double mx = chords.maxCoeff();

        std::cout << name << ","
                  << A.cols() << ","
                  << A.rows() << ","
                  << extra_rows << ","
                  << tight_rank << ","
                  << p10 << ","
                  << p50 << ","
                  << p90 << ","
                  << mx << "\n";
    }
}

} // namespace naja::cli::sample



