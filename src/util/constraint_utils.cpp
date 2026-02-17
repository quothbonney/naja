#include "util/constraint_utils.h"

#include <stdexcept>
#include <vector>

namespace naja::util {

int tight_constraint_rank(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const Eigen::VectorXd& x, double tol) {
    if (A.rows() != b.size()) {
        throw std::invalid_argument("tight_constraint_rank: A.rows != b.size");
    }
    if (A.cols() != x.size()) {
        throw std::invalid_argument("tight_constraint_rank: A.cols != x.size");
    }
    if (!(tol >= 0.0)) {
        throw std::invalid_argument("tight_constraint_rank: tol must be >= 0");
    }

    const Eigen::VectorXd slack = b - A * x;
    std::vector<int> idx;
    idx.reserve((size_t)A.rows());
    for (int i = 0; i < slack.size(); ++i) {
        if (slack[i] <= tol) idx.push_back(i);
    }
    if (idx.empty()) return 0;

    Eigen::MatrixXd At(idx.size(), A.cols());
    for (int k = 0; k < (int)idx.size(); ++k) {
        At.row(k) = A.row(idx[k]);
    }
    return (int)At.fullPivLu().rank();
}

} // namespace naja::util


