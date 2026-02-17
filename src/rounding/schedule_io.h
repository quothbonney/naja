#pragma once

#include <Eigen/Dense>
#include <string>

namespace naja::rounding {

struct PairSchedule {
    Eigen::VectorXi i;
    Eigen::VectorXi j;
    Eigen::VectorXd c;
    Eigen::VectorXd s;
};

PairSchedule load_pair_schedule_csv(const std::string& path);
void write_pair_schedule_csv(const std::string& path, const PairSchedule& sched);

} // namespace naja::rounding


