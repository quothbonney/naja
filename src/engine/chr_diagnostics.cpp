#include "engine/chr_diagnostics.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace naja::engine {

ChrAxisChordSummary chr_axis_chords(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const Eigen::VectorXd& x0) {
    if (A.rows() != b.size()) {
        throw std::invalid_argument("chr_axis_chords: A.rows != b.size");
    }
    if (A.cols() != x0.size()) {
        throw std::invalid_argument("chr_axis_chords: A.cols != x0.size");
    }
    const int m = A.rows();
    const int n = A.cols();

    // slack = b - A*x0
    const Eigen::VectorXd slack = b - A * x0;

    ChrAxisChordSummary out;
    out.chord_len.resize(n);
    out.chord_len.setZero();

    for (int j = 0; j < n; ++j) {
        double inv_min = std::numeric_limits<double>::infinity();
        double inv_max = -std::numeric_limits<double>::infinity();

        for (int i = 0; i < m; ++i) {
            const double s = slack[i];
            const double ae = A(i, j);

            double inv_dist;
            if (s == 0.0) {
                if (ae > 0.0) inv_dist = std::numeric_limits<double>::infinity();
                else if (ae < 0.0) inv_dist = -std::numeric_limits<double>::infinity();
                else inv_dist = 0.0;
            } else {
                inv_dist = ae / s;
                if (std::isnan(inv_dist)) inv_dist = 0.0;
            }

            inv_min = std::min(inv_min, inv_dist);
            inv_max = std::max(inv_max, inv_dist);
        }

        // chord endpoints in the kernel's alpha parameterization
        const double a_lo = 1.0 / inv_min;
        const double a_hi = 1.0 / inv_max;
        const double chord = a_hi - a_lo;
        out.chord_len[j] = chord;
    }

    return out;
}

} // namespace naja::engine



