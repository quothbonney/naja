#include "validate/bounds_check.h"

#include <algorithm>
#include <cmath>

namespace naja::validate {

BoundsResult check_bounds_backmap(
    const Eigen::MatrixXf& samples,
    const Eigen::MatrixXd& T,
    const Eigen::VectorXd& shift,
    const Eigen::VectorXd& lb,
    const Eigen::VectorXd& ub,
    int max_samples) {

    const Eigen::Index n_total = samples.cols();
    const Eigen::Index n_rxns = T.rows();
    const Eigen::Index step = std::max<Eigen::Index>(1, n_total / std::max(1, max_samples));
    const double eps = 1e-6;

    BoundsResult result{0, 0, 0.0};

    for (Eigen::Index j = 0; j < n_total; j += step) {
        // Backmap: v = T * y + shift
        Eigen::VectorXd y = samples.col(j).cast<double>();
        Eigen::VectorXd v = T * y + shift;

        for (Eigen::Index r = 0; r < n_rxns; ++r) {
            ++result.n_checked;
            double viol = std::max(lb[r] - v[r] - eps, v[r] - ub[r] - eps);
            if (viol > 0.0) {
                ++result.n_violations;
                result.worst_violation = std::max(result.worst_violation, viol);
            }
        }
    }

    return result;
}

} // namespace naja::validate

