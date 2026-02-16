#include <Eigen/Dense>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "csv_loader.h"
#include "pipeline/feasible_start_files.h"
#include "pipeline/model_contract.h"
#include "utils.h"

static void write_text(const std::string& path, const std::string& s) {
    std::ofstream f(path);
    if (!f.is_open()) throw std::runtime_error("cannot write: " + path);
    f << s;
}

int main() {
    // Minimal model dir: rounding + gem must exist for parse/contract patterns.
    // This test only exercises the feasible-start writer; it doesn't run sampling.
    const std::string root = "/tmp/naja_test_feasible_start_files";
    std::filesystem::remove_all(root);
    ensure_dir(root);
    const std::string model_dir = root + "/m";
    ensure_dir(model_dir);
    ensure_dir(model_dir + "/rounding");
    ensure_dir(model_dir + "/gem");

    naja::pipeline::ModelContract c = naja::pipeline::parse_model_dir(model_dir);
    const std::string p = c.rounding_dir + "/" + c.model_name + "_rounding";

    // Base constraints: -1 <= x0,x1 <= 1 in R^2
    write_text(p + "_A.csv", "1,0\n0,1\n-1,0\n0,-1\n");
    write_text(p + "_b.csv", "1\n1\n1\n1\n");

    // Infeasible inherited start: x0=2, x1=0
    write_text(p + "_start.csv", "2\n0\n");

    // Extra constraint: x0 <= 0.5
    write_text(p + "_extra_A.csv", "1,0\n");
    write_text(p + "_extra_b.csv", "0.5\n");

    // Dummy backmap files (contract isn't checked here but keep them nonempty)
    write_text(p + "_T.csv", "1,0\n");
    write_text(p + "_shift.csv", "0\n");

    // Dummy GEM files (not used here)
    write_text(c.gem_dir + "/reaction_ids.txt", "r0\n");
    write_text(c.gem_dir + "/l_bounds.csv", "-1\n");
    write_text(c.gem_dir + "/u_bounds.csv", "1\n");

    naja::pipeline::ensure_feasible_rounding_start_if_extra_present(c);

    Eigen::MatrixXd A = csv::loadMatrix(p + "_A.csv");
    Eigen::VectorXd b = csv::loadVector(p + "_b.csv");
    Eigen::MatrixXd Aex = csv::loadMatrix(p + "_extra_A.csv");
    Eigen::VectorXd bex = csv::loadVector(p + "_extra_b.csv");
    Eigen::VectorXd x = csv::loadVector(p + "_start.csv");

    Eigen::MatrixXd A_aug(A.rows() + Aex.rows(), A.cols());
    A_aug.topRows(A.rows()) = A;
    A_aug.bottomRows(Aex.rows()) = Aex;
    Eigen::VectorXd b_aug(b.size() + bex.size());
    b_aug.head(b.size()) = b;
    b_aug.tail(bex.size()) = bex;

    Eigen::VectorXd slack = (A_aug * x) - b_aug;
    if (slack.maxCoeff() > 1e-9) throw std::runtime_error("start still infeasible");
    if (x[0] > 0.5 + 1e-7) throw std::runtime_error("did not respect tightened bound");

    return 0;
}


