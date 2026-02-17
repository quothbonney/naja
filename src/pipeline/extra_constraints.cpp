#include "pipeline/extra_constraints.h"

#include <algorithm>
#include <stdexcept>

namespace naja::pipeline {

ExtraConstraintsMode parse_extra_constraints_mode(const std::string& s) {
    std::string t = s;
    std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    if (t.empty() || t == "auto") return ExtraConstraintsMode::Auto;
    if (t == "ignore" || t == "off" || t == "false" || t == "0") return ExtraConstraintsMode::Ignore;
    if (t == "require" || t == "on" || t == "true" || t == "1") return ExtraConstraintsMode::Require;
    throw std::invalid_argument("invalid extra constraints mode: " + s);
}

bool maybe_augment_extra_constraints(Eigen::MatrixXd& A,
                                    Eigen::VectorXd& b,
                                    const Eigen::MatrixXd* A_extra,
                                    const Eigen::VectorXd* b_extra,
                                    ExtraConstraintsMode mode,
                                    double extra_eps) {
    if (mode == ExtraConstraintsMode::Ignore) {
        return false;
    }

    const bool has = (A_extra != nullptr) || (b_extra != nullptr);
    if (!has) {
        if (mode == ExtraConstraintsMode::Require) {
            throw std::runtime_error("extra constraints required but not provided");
        }
        return false;
    }
    if (A_extra == nullptr || b_extra == nullptr) {
        throw std::runtime_error("incomplete extra constraint matrices (need both A_extra and b_extra)");
    }
    if (A_extra->rows() != b_extra->size()) {
        throw std::runtime_error("extra constraint matrices have mismatched dimensions");
    }
    if (A_extra->cols() != A.cols()) {
        throw std::runtime_error("extra constraint matrix has wrong number of columns");
    }

    Eigen::VectorXd b2 = *b_extra;
    if (extra_eps > 0.0) {
        b2.array() += extra_eps;
    }

    Eigen::MatrixXd A_aug(A.rows() + A_extra->rows(), A.cols());
    A_aug.topRows(A.rows()) = A;
    A_aug.bottomRows(A_extra->rows()) = *A_extra;
    Eigen::VectorXd b_aug(b.size() + b2.size());
    b_aug.head(b.size()) = b;
    b_aug.tail(b2.size()) = b2;
    A = std::move(A_aug);
    b = std::move(b_aug);
    return true;
}

} // namespace naja::pipeline


