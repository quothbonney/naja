#include "cli/validate_cli.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "csv_loader.h"
#include "npy.h"
#include "pipeline/model_contract.h"
#include "utils.h"
#include "validate/bounds_check.h"
#include "validate/chord_lengths.h"
#include "validate/mcmc.h"

namespace {

[[noreturn]] void die(const std::string& msg) {
    std::cerr << "error: " << msg << "\n\n";
    std::cerr << "usage:\n";
    std::cerr << "  naja validate --samples <file.npy> [--model-dir <dir>] [--n-chains 4]\n";
    std::exit(2);
}

std::string next_arg(int& i, int argc, char** argv, const std::string& flag) {
    if (i + 1 >= argc) die("missing value for " + flag);
    return std::string(argv[++i]);
}

// Write a JSON array of floats, compact
void json_array(std::ostream& o, const Eigen::VectorXf& v, int max_elements = 0) {
    o << "[";
    int n = (max_elements > 0 && max_elements < v.size()) ? max_elements : v.size();
    for (int i = 0; i < n; ++i) {
        if (i) o << ",";
        o << std::setprecision(4) << v[i];
    }
    if (max_elements > 0 && max_elements < v.size()) o << ",...";
    o << "]";
}

} // namespace

int naja_validate_cli_main(int argc, char** argv) {
    std::string samples_path;
    std::string model_dir;
    int n_chains = 4;

    for (int i = 0; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--samples") samples_path = next_arg(i, argc, argv, a);
        else if (a == "--model-dir") model_dir = next_arg(i, argc, argv, a);
        else if (a == "--n-chains") n_chains = std::stoi(next_arg(i, argc, argv, a));
        else die("unknown flag: " + a);
    }
    if (samples_path.empty()) die("missing --samples");
    if (n_chains < 1) die("--n-chains must be >= 1");

    // Load samples — detect dtype and shape from npy header
    std::cerr << "loading " << samples_path << "..." << std::endl;

    Eigen::MatrixXf samples;
    {
        std::ifstream file(samples_path, std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("cannot open: " + samples_path);

        char magic[6]; file.read(magic, 6);
        char ver[2]; file.read(ver, 2);
        uint16_t header_len; file.read(reinterpret_cast<char*>(&header_len), 2);
        std::string header(header_len, ' ');
        file.read(&header[0], header_len);

        // Parse dtype
        bool is_f32 = header.find("'<f4'") != std::string::npos;
        bool is_f64 = header.find("'<f8'") != std::string::npos;
        if (!is_f32 && !is_f64) throw std::runtime_error("unsupported dtype in " + samples_path);

        // Parse shape
        auto sp = header.find("'shape':");
        auto p1 = header.find('(', sp);
        auto p2 = header.find(')', p1);
        std::string shape_str = header.substr(p1 + 1, p2 - p1 - 1);
        std::replace(shape_str.begin(), shape_str.end(), ',', ' ');
        std::stringstream ss(shape_str);
        int rows, cols;
        ss >> rows >> cols;

        // Read into (dim, n_samples) col-major float32
        // Heuristic: smaller dim is the reduced dimension
        int file_dim = std::min(rows, cols);
        int file_nsamp = std::max(rows, cols);
        samples.resize(file_dim, file_nsamp);

        if (is_f32) {
            // C-order (rows, cols) float32 → read directly
            // If rows < cols: file is (dim, nsamp), C-order = row-major
            //   Eigen col-major (dim, nsamp): col k = samples[:, k] is contiguous
            //   C-order row k = file row k is contiguous
            //   We need: samples(d, s) = file[d, s] = data[d * cols + s] (if rows=dim)
            //            OR file[s, d] = data[s * cols + d] (if rows=nsamp, cols=dim)
            // Simplest: read into buffer, interpret correctly
            std::vector<float> buf(static_cast<size_t>(rows) * cols);
            file.read(reinterpret_cast<char*>(buf.data()), buf.size() * sizeof(float));

            if (rows <= cols) {
                // file is (dim, nsamp) C-order — row d has all samples for dim d
                for (int d = 0; d < file_dim; ++d)
                    for (int s = 0; s < file_nsamp; ++s)
                        samples(d, s) = buf[d * file_nsamp + s];
            } else {
                // file is (nsamp, dim) C-order — row s has all dims for sample s
                for (int s = 0; s < file_nsamp; ++s)
                    for (int d = 0; d < file_dim; ++d)
                        samples(d, s) = buf[s * file_dim + d];
            }
        } else {
            // float64 — read and cast
            std::vector<double> buf(static_cast<size_t>(rows) * cols);
            file.read(reinterpret_cast<char*>(buf.data()), buf.size() * sizeof(double));

            if (rows <= cols) {
                for (int d = 0; d < file_dim; ++d)
                    for (int s = 0; s < file_nsamp; ++s)
                        samples(d, s) = static_cast<float>(buf[d * file_nsamp + s]);
            } else {
                for (int s = 0; s < file_nsamp; ++s)
                    for (int d = 0; d < file_dim; ++d)
                        samples(d, s) = static_cast<float>(buf[s * file_dim + d]);
            }
        }
    }

    const int dim = samples.rows();
    const int n_total = samples.cols();
    const int spc = n_total / n_chains;

    std::cerr << "  shape: (" << dim << ", " << n_total << ") = " << n_chains << " chains x " << spc << " samples" << std::endl;

    if (n_total % n_chains != 0) {
        throw std::runtime_error("n_total (" + std::to_string(n_total) + ") not divisible by n_chains (" + std::to_string(n_chains) + ")");
    }

    // Active dimensions (nonzero variance)
    Eigen::VectorXf var = Eigen::VectorXf::Zero(dim);
    for (int d = 0; d < dim; ++d) {
        float mean = samples.row(d).mean();
        var[d] = (samples.row(d).array() - mean).square().mean();
    }
    int active_dims = (var.array() > 1e-20f).count();

    // ESS
    std::cerr << "  computing ESS..." << std::endl;
    auto ess = naja::validate::compute_ess(samples, n_chains);

    // Split-R-hat
    std::cerr << "  computing split-R-hat..." << std::endl;
    auto rhat = naja::validate::compute_split_rhat(samples, n_chains);

    // Bounds check (optional, needs model dir)
    bool has_bounds = false;
    naja::validate::BoundsResult bounds{0, 0, 0.0};
    naja::validate::ChordSummary chords;
    bool has_chords = false;

    if (!model_dir.empty()) {
        auto contract = naja::pipeline::parse_model_dir(model_dir);
        std::string T_path = contract.rounding_dir + "/" + contract.model_name + "_rounding_T.csv";
        std::string shift_path = contract.rounding_dir + "/" + contract.model_name + "_rounding_shift.csv";
        std::string lb_path = contract.gem_dir + "/l_bounds.csv";
        std::string ub_path = contract.gem_dir + "/u_bounds.csv";
        std::string A_path = contract.rounding_dir + "/" + contract.model_name + "_rounding_A.csv";
        std::string b_path = contract.rounding_dir + "/" + contract.model_name + "_rounding_b.csv";
        std::string start_path = contract.rounding_dir + "/" + contract.model_name + "_rounding_start.csv";

        if (path_exists(T_path) && path_exists(lb_path)) {
            std::cerr << "  checking bounds compliance..." << std::endl;
            Eigen::MatrixXd T = csv::loadMatrix(T_path);
            Eigen::VectorXd shift = csv::loadVector(shift_path);
            Eigen::VectorXd lb = csv::loadVector(lb_path);
            Eigen::VectorXd ub = csv::loadVector(ub_path);
            bounds = naja::validate::check_bounds_backmap(samples, T, shift, lb, ub);
            has_bounds = true;
        }

        if (path_exists(A_path) && path_exists(b_path) && path_exists(start_path)) {
            Eigen::MatrixXd A = csv::loadMatrix(A_path);
            Eigen::VectorXd b = csv::loadVector(b_path);
            Eigen::VectorXd x0 = csv::loadVector(start_path);

            // Load extra constraints if present
            std::string eA_path = contract.rounding_dir + "/" + contract.model_name + "_rounding_extra_A.csv";
            std::string eb_path = contract.rounding_dir + "/" + contract.model_name + "_rounding_extra_b.csv";
            if (path_exists(eA_path) && path_exists(eb_path)) {
                Eigen::MatrixXd Aex = csv::loadMatrix(eA_path);
                Eigen::VectorXd bex = csv::loadVector(eb_path);
                Eigen::MatrixXd A_aug(A.rows() + Aex.rows(), A.cols());
                A_aug.topRows(A.rows()) = A;
                A_aug.bottomRows(Aex.rows()) = Aex;
                Eigen::VectorXd b_aug(b.size() + bex.size());
                b_aug.head(b.size()) = b;
                b_aug.tail(bex.size()) = bex;
                A = std::move(A_aug);
                b = std::move(b_aug);
            }

            chords = naja::validate::chr_axis_chords(A, b, x0);
            has_chords = true;
        }
    }

    // Determine status
    std::string status = "OK";
    if (rhat.above_1_2 > dim / 4) status = "FAIL";
    else if (rhat.above_1_1 > dim / 4 || ess.min < 10) status = "WARN";

    // One-line summary to stderr
    std::cerr << "\n[" << status << "] "
              << "ESS min=" << std::setprecision(0) << std::fixed << ess.min
              << " p10=" << ess.p10
              << " med=" << ess.median
              << "  R-hat med=" << std::setprecision(3) << rhat.median
              << " max=" << rhat.max
              << " >1.1=" << rhat.above_1_1
              << " >1.2=" << rhat.above_1_2;
    if (has_bounds) {
        std::cerr << "  bounds=" << bounds.n_violations << "/" << bounds.n_checked;
    }
    std::cerr << std::endl;

    // JSON to stdout
    std::cout << "{\n";
    std::cout << "  \"samples_file\": \"" << samples_path << "\",\n";
    std::cout << "  \"shape\": [" << n_total << ", " << dim << "],\n";
    std::cout << "  \"n_chains\": " << n_chains << ",\n";
    std::cout << "  \"n_samples_per_chain\": " << spc << ",\n";
    std::cout << "  \"active_dims\": " << active_dims << ",\n";
    std::cout << "  \"status\": \"" << status << "\",\n";

    std::cout << "  \"ess\": {\n";
    std::cout << "    \"min\": " << std::setprecision(1) << std::fixed << ess.min << ",\n";
    std::cout << "    \"p10\": " << ess.p10 << ",\n";
    std::cout << "    \"median\": " << ess.median << ",\n";
    std::cout << "    \"max\": " << ess.max << "\n";
    std::cout << "  },\n";

    std::cout << "  \"rhat\": {\n";
    std::cout << "    \"median\": " << std::setprecision(4) << rhat.median << ",\n";
    std::cout << "    \"max\": " << rhat.max << ",\n";
    std::cout << "    \"above_1_1\": " << rhat.above_1_1 << ",\n";
    std::cout << "    \"above_1_2\": " << rhat.above_1_2 << "\n";
    std::cout << "  },\n";

    if (has_bounds) {
        std::cout << "  \"bounds\": {\n";
        std::cout << "    \"violations\": " << bounds.n_violations << ",\n";
        std::cout << "    \"checked\": " << bounds.n_checked << ",\n";
        std::cout << "    \"worst\": " << std::setprecision(6) << bounds.worst_violation << "\n";
        std::cout << "  },\n";
    }

    if (has_chords) {
        int pos_chords = (chords.chord_len.array() > 0).count();
        double chord_min = (pos_chords > 0) ? chords.chord_len.array().max(0).minCoeff() : 0;
        // min of positive values
        chord_min = 1e30;
        for (int i = 0; i < chords.chord_len.size(); ++i) {
            if (chords.chord_len[i] > 0 && chords.chord_len[i] < chord_min)
                chord_min = chords.chord_len[i];
        }
        if (pos_chords == 0) chord_min = 0;

        std::cout << "  \"chords\": {\n";
        std::cout << "    \"positive_dims\": " << pos_chords << ",\n";
        std::cout << "    \"min_positive\": " << std::setprecision(6) << chord_min << ",\n";
        std::cout << "    \"max\": " << chords.chord_len.maxCoeff() << "\n";
        std::cout << "  },\n";
    }

    // Per-dim arrays (full)
    std::cout << "  \"per_dim_ess\": ";
    json_array(std::cout, ess.ess);
    std::cout << ",\n";

    std::cout << "  \"per_dim_rhat\": ";
    json_array(std::cout, rhat.rhat);
    std::cout << "\n";

    std::cout << "}\n";

    return 0;
}

