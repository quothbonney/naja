#include "pipeline/feasible_start_files.h"

#include <Eigen/Dense>

#include <filesystem>
#include <exception>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

#include "csv_loader.h"
#include "util/start_feasibility.h"
#include "pipeline/feasible_start_lp.h"
#include "utils.h"

namespace naja::pipeline {
namespace {

constexpr double kPrepareStartFeasibilityEps = 1e-7;

static void write_vector_csv(const std::string& path, const Eigen::VectorXd& v) {
    std::ofstream f(path);
    if (!f.is_open()) throw std::runtime_error("cannot write: " + path);
    for (int i = 0; i < v.size(); ++i) {
        // Round-trip-safe for double.
        f << std::setprecision(17) << v[i] << "\n";
    }
}

static void remove_path_if_exists(const std::string& path) {
    if (!path_exists(path)) return;
    std::error_code ec;
    std::filesystem::remove(path, ec);
    if (ec) {
        throw std::runtime_error("cannot remove: " + path);
    }
}

} // namespace

void ensure_feasible_rounding_start_if_extra_present(const ModelContract& c) {
    const std::string prefix = c.rounding_dir + "/" + c.model_name + "_rounding";
    const std::string A_path = prefix + "_A.csv";
    const std::string b_path = prefix + "_b.csv";
    const std::string start_path = prefix + "_start.csv";
    const std::string extra_A = prefix + "_extra_A.csv";
    const std::string extra_b = prefix + "_extra_b.csv";

    bool has_extra_A = path_exists(extra_A);
    bool has_extra_b = path_exists(extra_b);
    if (!has_extra_A && !has_extra_b) return;
    if (has_extra_A != has_extra_b) {
        throw std::runtime_error("extra constraints must be both-present or both-absent: " + extra_A + " / " + extra_b);
    }

    Eigen::MatrixXd A = csv::loadMatrix(A_path);
    Eigen::VectorXd b = csv::loadVector(b_path);
    Eigen::MatrixXd Aex = csv::loadMatrix(extra_A);
    Eigen::VectorXd bex = csv::loadVector(extra_b);
    if (Aex.rows() != bex.size()) throw std::runtime_error("extra constraint matrices have mismatched dimensions");
    if (Aex.cols() != A.cols()) throw std::runtime_error("extra constraint matrix has wrong number of columns");

    Eigen::MatrixXd A_aug(A.rows() + Aex.rows(), A.cols());
    A_aug.topRows(A.rows()) = A;
    A_aug.bottomRows(Aex.rows()) = Aex;
    Eigen::VectorXd b_aug(b.size() + bex.size());
    b_aug.head(b.size()) = b;
    b_aug.tail(bex.size()) = bex;

    // Fast path: if the current start already satisfies augmented constraints,
    // keep it and skip the LP solve.
    if (path_exists(start_path)) {
        try {
            Eigen::VectorXd x_existing = csv::loadVector(start_path);
            if (x_existing.size() == A_aug.cols()) {
                naja::util::require_feasible_start(A_aug, b_aug, x_existing, kPrepareStartFeasibilityEps, "feasible_start_files(existing)");
                return;
            }
        } catch (const std::exception&) {
            // Fall through to LP recomputation.
        }
    }

    auto [x, r] = axis_aligned_cube_center_lp_gurobi(A_aug, b_aug);
    (void)r;
    naja::util::require_feasible_start(A_aug, b_aug, x, kPrepareStartFeasibilityEps, "feasible_start_files");

    // Important: the current start file may be a symlink (symlink mode). Remove it first.
    remove_path_if_exists(start_path);
    write_vector_csv(start_path, x);
}

} // namespace naja::pipeline
