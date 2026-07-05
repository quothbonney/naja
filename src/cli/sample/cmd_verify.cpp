#include "cli/sample/commands.h"

#include <Eigen/Dense>

#include <iostream>
#include <stdexcept>
#include <string>

#include "cli/sample/common.h"
#include "pipeline/model_contract.h"
#include "pipeline/model_io.h"
#include "utils.h"

namespace naja::cli::sample {

void cmd_verify(int argc, char** argv) {
    std::string model_dir;
    bool backmap = false;

    for (int i = 0; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model-dir") model_dir = next_arg(i, argc, argv, a);
        else if (a == "--backmap") backmap = true;
        else die_usage("unknown flag: " + a);
    }
    if (model_dir.empty()) die_usage("missing --model-dir");

    naja::pipeline::ModelContract c = naja::pipeline::parse_model_dir(model_dir);
    naja::pipeline::validate_contract(c, backmap);

    // Load through RoundingReader so verification is layout-agnostic (bundle or
    // legacy CSV) and reflects the actual arrays that sampling would consume.
    naja::pipeline::RoundingReader reader(c.rounding_dir, c.model_name);
    const bool bundle = reader.is_bundle();

    Eigen::MatrixXd A = reader.A();
    Eigen::VectorXd b = reader.b();
    Eigen::VectorXd start = reader.start();
    if (b.size() != A.rows()) throw std::runtime_error("b rows != A rows");
    if (start.size() != A.cols()) throw std::runtime_error("start dim != A cols");

    int extra_rows = 0;
    bool extra_present = reader.has_extra();
    if (extra_present) {
        Eigen::MatrixXd eA = reader.extra_A();
        Eigen::VectorXd eb = reader.extra_b();
        extra_rows = static_cast<int>(eA.rows());
        if (eA.rows() != eb.size()) throw std::runtime_error("extra_A rows != extra_b rows");
        if (eA.cols() != A.cols()) throw std::runtime_error("extra_A cols != A cols");
    }

    int rxn_ids_n = 0;
    int lb_n = 0;
    int ub_n = 0;
    bool gem_ok = false;
    bool gem_exists = is_directory(c.gem_dir);
    if (gem_exists) {
        const std::string rxn_ids = c.gem_dir + "/reaction_ids.txt";
        const std::string lb = c.gem_dir + "/l_bounds.csv";
        const std::string ub = c.gem_dir + "/u_bounds.csv";
        if (path_exists(rxn_ids) && path_exists(lb) && path_exists(ub)) {
            rxn_ids_n = naja::pipeline::text_line_count(rxn_ids);
            lb_n = naja::pipeline::csv_shape(lb).rows;
            ub_n = naja::pipeline::csv_shape(ub).rows;
            gem_ok = (rxn_ids_n == lb_n) && (rxn_ids_n == ub_n);
        }
    }

    std::cout << "OK\n";
    std::cout << "model_dir           :: " << c.model_dir << "\n";
    std::cout << "model_name          :: " << c.model_name << "\n";
    std::cout << "rounding_layout     :: " << (bundle ? "bundle" : "legacy_csv") << "\n";
    std::cout << "A                   :: " << A.rows() << " x " << A.cols() << "\n";
    std::cout << "b                   :: " << b.size() << "\n";
    std::cout << "start               :: " << start.size() << "\n";
    if (backmap) {
        if (!reader.has_backmap()) throw std::runtime_error("backmap requested but T/shift absent");
        Eigen::MatrixXd T = reader.T();
        Eigen::VectorXd shift = reader.shift();
        if (T.cols() != A.cols()) throw std::runtime_error("T cols != A cols");
        if (shift.size() != T.rows()) throw std::runtime_error("shift dim != T rows");
        std::cout << "T                   :: " << T.rows() << " x " << T.cols() << "\n";
        std::cout << "shift               :: " << shift.size() << "\n";
    }
    std::cout << "extra_constraints   :: " << (extra_present ? "present" : "absent") << "\n";
    if (extra_present) std::cout << "extra_rows          :: " << extra_rows << "\n";
    std::cout << "gem_dir             :: " << (gem_exists ? "present" : "absent") << "\n";
    if (gem_exists) {
        std::cout << "gem_counts          :: rxn_ids=" << rxn_ids_n << " lb=" << lb_n << " ub=" << ub_n << "\n";
        std::cout << "gem_consistent      :: " << (gem_ok ? "yes" : "no") << "\n";
    }
}

} // namespace naja::cli::sample
