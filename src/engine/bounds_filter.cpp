#include "engine/bounds_filter.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "csv_loader.h"
#include "npy.h"
#include "pipeline/text_io.h"
#include "utils.h"

namespace naja::engine {
namespace {

void write_bounds_report_json(const std::string& path,
                              const RuntimeConfig& cfg,
                              double eps,
                              int n_total,
                              int n_valid,
                              double mean_violation,
                              double p99_violation,
                              double max_violation,
                              const std::vector<std::pair<int, double>>& top_rxn,
                              const std::vector<std::string>& rxn_ids) {
    std::ofstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot write bounds report: " + path);
    }
    f << "{\n";
    f << "  \"model\": \"" << cfg.MODEL_NAME << "\",\n";
    f << "  \"model_dir\": \"" << make_absolute_path(cfg.MODEL_DIR) << "\",\n";
    f << "  \"run_dir\": \"" << make_absolute_path(cfg.OUT_DIR) << "\",\n";
    f << "  \"eps\": " << std::setprecision(12) << eps << ",\n";
    f << "  \"n_total\": " << n_total << ",\n";
    f << "  \"n_valid\": " << n_valid << ",\n";
    f << "  \"n_invalid\": " << (n_total - n_valid) << ",\n";
    f << "  \"invalid_fraction\": " << std::setprecision(12)
      << (n_total ? (double)(n_total - n_valid) / (double)n_total : 0.0) << ",\n";
    f << "  \"mean_violation\": " << std::setprecision(12) << mean_violation << ",\n";
    f << "  \"p99_violation\": " << std::setprecision(12) << p99_violation << ",\n";
    f << "  \"max_violation\": " << std::setprecision(12) << max_violation << ",\n";
    f << "  \"outputs\": {\n";
    f << "    \"valid_mask\": \"" << make_absolute_path(cfg.OUT_DIR + "/valid_mask.npy") << "\",\n";
    f << "    \"samples_valid\": " << (cfg.WRITE_SAMPLES_VALID ? ("\"" + make_absolute_path(cfg.OUT_DIR + "/samples_valid.npy") + "\"") : "null") << "\n";
    f << "  },\n";
    f << "  \"top_violations\": [\n";
    for (size_t i = 0; i < top_rxn.size(); ++i) {
        int idx = top_rxn[i].first;
        double v = top_rxn[i].second;
        f << "    {\"index\": " << idx
          << ", \"reaction\": \"" << (idx >= 0 && idx < (int)rxn_ids.size() ? rxn_ids[idx] : "") << "\""
          << ", \"worst_violation\": " << std::setprecision(12) << v << "}";
        if (i + 1 != top_rxn.size()) f << ",";
        f << "\n";
    }
    f << "  ]\n";
    f << "}\n";
}

} // namespace

void bounds_filter_and_write(const RuntimeConfig& cfg, const Eigen::MatrixXd& samples_out) {
    if (!cfg.BACK_TRANSFORM) {
        throw std::runtime_error("bounds filtering requires BACK_TRANSFORM=true (samples must be in GEM/original space)");
    }

    const std::string gem_dir = cfg.MODEL_DIR + "/gem";
    const std::string rxn_ids_path = gem_dir + "/reaction_ids.txt";
    const std::string lb_path = gem_dir + "/l_bounds.csv";
    const std::string ub_path = gem_dir + "/u_bounds.csv";

    ::require_nonempty_file(rxn_ids_path, "gem/reaction_ids.txt");
    ::require_nonempty_file(lb_path, "gem/l_bounds.csv");
    ::require_nonempty_file(ub_path, "gem/u_bounds.csv");

    std::vector<std::string> rxn_ids = naja::pipeline::read_nonempty_trimmed_lines(rxn_ids_path);
    Eigen::VectorXd lb = csv::loadVector(lb_path);
    Eigen::VectorXd ub = csv::loadVector(ub_path);

    const int dim = (int)samples_out.rows();
    const int n_total = (int)samples_out.cols();
    if ((int)rxn_ids.size() != dim) {
        throw std::runtime_error("reaction_ids length mismatch: expected " + std::to_string(dim) + " got " + std::to_string(rxn_ids.size()));
    }
    if (lb.size() != dim || ub.size() != dim) {
        throw std::runtime_error("bounds length mismatch: expected " + std::to_string(dim) + " got lb=" + std::to_string(lb.size()) + " ub=" + std::to_string(ub.size()));
    }

    std::vector<uint8_t> valid_mask((size_t)n_total, 1);
    std::vector<double> per_sample_violation((size_t)n_total, 0.0);
    double max_violation = 0.0;
    Eigen::VectorXd worst_by_rxn = Eigen::VectorXd::Zero(dim);

    const double eps = cfg.BOUNDS_EPS;
    for (int j = 0; j < n_total; ++j) {
        bool ok = true;
        double worst = 0.0;
        for (int i = 0; i < dim; ++i) {
            double x = samples_out(i, j);
            double v_lo = (lb(i) - x) - eps;
            double v_hi = (x - ub(i)) - eps;
            double v = (v_lo > v_hi) ? v_lo : v_hi;
            if (v > 0.0) {
                ok = false;
                if (v > max_violation) max_violation = v;
                if (v > worst_by_rxn(i)) worst_by_rxn(i) = v;
                if (v > worst) worst = v;
            }
        }
        valid_mask[(size_t)j] = ok ? 1 : 0;
        per_sample_violation[(size_t)j] = worst;
    }

    int n_valid = 0;
    for (uint8_t b : valid_mask) n_valid += (b != 0);

    npy::save_u8_1d(cfg.OUT_DIR + "/valid_mask.npy", valid_mask);

    {
        const double valid_fraction = n_total ? ((double)n_valid / (double)n_total) : 0.0;
        std::ofstream vf(cfg.OUT_DIR + "/valid_fraction.txt");
        if (!vf.is_open()) {
            throw std::runtime_error("cannot write valid_fraction.txt");
        }
        vf << std::setprecision(12) << valid_fraction << "\n";
    }

    if (cfg.WRITE_SAMPLES_VALID) {
        Eigen::MatrixXd samples_valid(dim, n_valid);
        int k = 0;
        for (int j = 0; j < n_total; ++j) {
            if (valid_mask[(size_t)j]) {
                samples_valid.col(k++) = samples_out.col(j);
            }
        }
        npy::save(cfg.OUT_DIR + "/samples_valid.npy", samples_valid);
    }

    // Summary stats over per-sample violation (max across dims, already thresholded).
    double mean_violation = 0.0;
    for (double v : per_sample_violation) mean_violation += v;
    mean_violation = n_total ? (mean_violation / (double)n_total) : 0.0;

    double p99_violation = 0.0;
    if (n_total > 0) {
        size_t k = (size_t)((99 * (n_total - 1)) / 100);
        std::nth_element(per_sample_violation.begin(),
                         per_sample_violation.begin() + (ptrdiff_t)k,
                         per_sample_violation.end());
        p99_violation = per_sample_violation[k];
    }

    const int k_max = std::min(dim, 10);
    std::vector<std::pair<int, double>> top;
    top.reserve((size_t)k_max);
    for (int i = 0; i < dim; ++i) {
        top.emplace_back(i, worst_by_rxn(i));
    }
    std::partial_sort(top.begin(), top.begin() + k_max, top.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });
    top.resize((size_t)k_max);
    while (!top.empty() && top.back().second <= 0.0) top.pop_back();

    write_bounds_report_json(cfg.OUT_DIR + "/bounds_report.json",
                             cfg,
                             eps,
                             n_total,
                             n_valid,
                             mean_violation,
                             p99_violation,
                             max_violation,
                             top,
                             rxn_ids);
}

}
