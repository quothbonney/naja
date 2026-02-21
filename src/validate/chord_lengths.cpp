#include "validate/chord_lengths.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace naja::validate {

ChordSummary chr_axis_chords(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const Eigen::VectorXd& x0) {
    if (A.rows() != b.size()) throw std::invalid_argument("chr_axis_chords: A.rows != b.size");
    if (A.cols() != x0.size()) throw std::invalid_argument("chr_axis_chords: A.cols != x0.size");

    const int m = A.rows();
    const int n = A.cols();
    const Eigen::VectorXd slack = b - A * x0;

    ChordSummary out;
    out.chord_len.resize(n);

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
        out.chord_len[j] = (1.0 / inv_max) - (1.0 / inv_min);
    }
    return out;
}

} // namespace naja::validate

