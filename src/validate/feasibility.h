#pragma once

#include <Eigen/Dense>

#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace naja::validate {

inline void require_feasible_start(const Eigen::MatrixXd& A,
                                   const Eigen::VectorXd& b,
                                   const Eigen::VectorXd& x0,
                                   double eps,
                                   const std::string& tag) {
    if (A.cols() != x0.size()) {
        throw std::runtime_error("dimension mismatch: A.cols != x0.size");
    }
    if (A.rows() != b.size()) {
        throw std::runtime_error("dimension mismatch: A.rows != b.size");
    }
    Eigen::VectorXd slack = (A * x0) - b;
    const double max_v = slack.maxCoeff();
    if (!(max_v <= eps)) {
        const int violated = (slack.array() > eps).count();
        std::ostringstream oss;
        oss << "infeasible start point";
        if (!tag.empty()) oss << " (" << tag << ")";
        oss << ": max(A*x0 - b)=" << std::setprecision(12) << max_v
            << " > eps=" << std::setprecision(12) << eps
            << ", violated=" << violated << "/" << slack.size();
        throw std::runtime_error(oss.str());
    }
}

} // namespace naja::validate

