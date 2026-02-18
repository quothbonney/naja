#include "pipeline/feasible_start_lp.h"

#include <gurobi_c++.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace naja::pipeline {

static void ensure_gurobi_license_env() {
    static std::once_flag once;
    std::call_once(once, []() {
        const char* existing = std::getenv("GRB_LICENSE_FILE");
        if (existing != nullptr && existing[0] != '\0') {
            return;
        }

        const char* path = "/data/rbg/users/jdcarson/gurobi.lic";
        if (!std::filesystem::exists(path)) {
            throw std::runtime_error(std::string("GRB_LICENSE_FILE is unset and license file does not exist: ") + path);
        }
        if (setenv("GRB_LICENSE_FILE", path, 1) != 0) {
            throw std::runtime_error("failed to set GRB_LICENSE_FILE");
        }
    });
}

std::pair<Eigen::VectorXd, double> axis_aligned_cube_center_lp_gurobi(const Eigen::MatrixXd& A,
                                                                      const Eigen::VectorXd& b) {
    if (A.rows() != b.size()) throw std::runtime_error("axis_aligned_cube_center: A.rows != b.size");
    const int m = A.rows();
    const int n = A.cols();
    if (n <= 0 || m <= 0) throw std::runtime_error("axis_aligned_cube_center: empty A");

    try {
        ensure_gurobi_license_env();
        GRBEnv env;
        env.set(GRB_IntParam_OutputFlag, 0);
        GRBModel model(env);
        model.set(GRB_IntAttr_ModelSense, GRB_MAXIMIZE);
        model.set(GRB_IntParam_NumericFocus, 3);
        model.set(GRB_DoubleParam_FeasibilityTol, 1e-9);
        model.set(GRB_DoubleParam_OptimalityTol, 1e-9);
        model.set(GRB_IntParam_Threads, 1);

        std::vector<GRBVar> x((size_t)n);
        for (int j = 0; j < n; ++j) {
            x[(size_t)j] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "x_" + std::to_string(j));
        }
        GRBVar r = model.addVar(0.0, GRB_INFINITY, 1.0, GRB_CONTINUOUS, "r");

        for (int i = 0; i < m; ++i) {
            double w = 0.0;
            for (int j = 0; j < n; ++j) {
                w += std::abs(A(i, j));
            }
            if (!std::isfinite(w)) throw std::runtime_error("axis_aligned_cube_center: non-finite row l1 norm");

            GRBLinExpr lhs = 0.0;
            for (int j = 0; j < n; ++j) {
                const double a = A(i, j);
                if (a != 0.0) lhs += a * x[(size_t)j];
            }
            lhs += w * r;
            if (!std::isfinite(b[i])) throw std::runtime_error("axis_aligned_cube_center: non-finite b");
            model.addConstr(lhs <= b[i], "c_" + std::to_string(i));
        }

        model.optimize();
        const int status = model.get(GRB_IntAttr_Status);
        if (status != GRB_OPTIMAL) {
            throw std::runtime_error("axis_aligned_cube_center: LP status " + std::to_string(status));
        }

        Eigen::VectorXd x_out(n);
        for (int j = 0; j < n; ++j) x_out[j] = x[(size_t)j].get(GRB_DoubleAttr_X);
        double r_out = r.get(GRB_DoubleAttr_X);
        return {x_out, r_out};
    } catch (const GRBException& e) {
        throw std::runtime_error("gurobi error " + std::to_string(e.getErrorCode()) + ": " + std::string(e.getMessage()));
    }
}

} // namespace naja::pipeline

