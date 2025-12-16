#include "cli/sample/commands.h"

#include <iostream>
#include <stdexcept>
#include <string>

#include "cli/sample/common.h"
#include "pipeline/model_contract.h"
#include "pipeline/verify_report.h"
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

    const std::string A_path = c.rounding_dir + "/" + c.model_name + "_rounding_A.csv";
    const std::string b_path = c.rounding_dir + "/" + c.model_name + "_rounding_b.csv";
    const std::string start_path = c.rounding_dir + "/" + c.model_name + "_rounding_start.csv";
    const std::string T_path = c.rounding_dir + "/" + c.model_name + "_rounding_T.csv";
    const std::string shift_path = c.rounding_dir + "/" + c.model_name + "_rounding_shift.csv";
    const std::string extra_A_path = c.rounding_dir + "/" + c.model_name + "_rounding_extra_A.csv";
    const std::string extra_b_path = c.rounding_dir + "/" + c.model_name + "_rounding_extra_b.csv";

    naja::pipeline::CsvShape A = naja::pipeline::csv_shape(A_path);
    naja::pipeline::CsvShape b = naja::pipeline::csv_shape(b_path);
    naja::pipeline::CsvShape start = naja::pipeline::csv_shape(start_path);
    if (b.rows != A.rows) throw std::runtime_error("b rows != A rows");
    if (start.rows != A.cols) throw std::runtime_error("start dim != A cols");

    int extra_rows = 0;
    bool extra_present = path_exists(extra_A_path);
    if (extra_present) {
        naja::pipeline::CsvShape extraA = naja::pipeline::csv_shape(extra_A_path);
        naja::pipeline::CsvShape extrab = naja::pipeline::csv_shape(extra_b_path);
        extra_rows = extraA.rows;
        if (extraA.rows != extrab.rows) throw std::runtime_error("extra_A rows != extra_b rows");
        if (extraA.cols != A.cols) throw std::runtime_error("extra_A cols != A cols");
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
    std::cout << "A                   :: " << A.rows << " x " << A.cols << "\n";
    std::cout << "b                   :: " << b.rows << " x " << b.cols << "\n";
    std::cout << "start               :: " << start.rows << " x " << start.cols << "\n";
    if (backmap) {
        naja::pipeline::CsvShape T = naja::pipeline::csv_shape(T_path);
        naja::pipeline::CsvShape shift = naja::pipeline::csv_shape(shift_path);
        if (T.cols != A.cols) throw std::runtime_error("T cols != A cols");
        if (shift.rows != T.rows) throw std::runtime_error("shift dim != T rows");
        std::cout << "T                   :: " << T.rows << " x " << T.cols << "\n";
        std::cout << "shift               :: " << shift.rows << " x " << shift.cols << "\n";
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


