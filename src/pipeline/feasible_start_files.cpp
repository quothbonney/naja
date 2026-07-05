#include "pipeline/feasible_start_files.h"

#include <Eigen/Dense>

#include <filesystem>
#include <exception>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

#include "csv_loader.h"
#include "npz.h"
#include "pipeline/model_io.h"
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
    RoundingReader reader(c.rounding_dir, c.model_name);

    // No KO delta => the shared base start is already feasible; nothing to do.
    if (!reader.has_extra()) return;

    Eigen::MatrixXd A = reader.A();
    Eigen::MatrixXd Aex = reader.extra_A();
    Eigen::VectorXd bex = reader.extra_b();
    Eigen::VectorXd b = reader.b();
    if (Aex.rows() != bex.size()) throw std::runtime_error("extra constraint matrices have mismatched dimensions");
    if (Aex.cols() != A.cols()) throw std::runtime_error("extra constraint matrix has wrong number of columns");

    Eigen::MatrixXd A_aug(A.rows() + Aex.rows(), A.cols());
    A_aug.topRows(A.rows()) = A;
    A_aug.bottomRows(Aex.rows()) = Aex;
    Eigen::VectorXd b_aug(b.size() + bex.size());
    b_aug.head(b.size()) = b;
    b_aug.tail(bex.size()) = bex;

    // Fast path: if the current start (sidecar, bundle, or legacy) already
    // satisfies the augmented constraints, keep it and skip the LP solve.
    try {
        Eigen::VectorXd x_existing = reader.start();
        if (x_existing.size() == A_aug.cols()) {
            naja::util::require_feasible_start(A_aug, b_aug, x_existing, kPrepareStartFeasibilityEps, "feasible_start_files(existing)");
            return;
        }
    } catch (const std::exception&) {
        // Fall through to LP recomputation.
    }

    auto [x, r] = axis_aligned_cube_center_lp_gurobi(A_aug, b_aug);
    (void)r;
    naja::util::require_feasible_start(A_aug, b_aug, x, kPrepareStartFeasibilityEps, "feasible_start_files");

    // Write to the layout-appropriate location. For bundles this is a per-model
    // start.npy sidecar; we must NOT mutate polytope.npz, which is shared across
    // KO variants by symlink. For legacy the target may itself be a symlink, so
    // remove it first before writing the concrete file.
    const std::string start_path = reader.start_write_path();
    remove_path_if_exists(start_path);
    if (reader.is_bundle()) {
        npz::save_npy_vector(start_path, x);
    } else {
        write_vector_csv(start_path, x);
    }
}

} // namespace naja::pipeline
