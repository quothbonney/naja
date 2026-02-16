#pragma once

#include <Eigen/Dense>
#include <string>

namespace naja::pipeline {

struct PairSchedule {
    Eigen::VectorXi i;
    Eigen::VectorXi j;
    Eigen::VectorXd c;
    Eigen::VectorXd s;
};

// CSV format: 4 columns per row: i,j,c,s (no header).
PairSchedule load_pair_schedule_csv(const std::string& path);
void write_pair_schedule_csv(const std::string& path, const PairSchedule& sched);

} // namespace naja::pipeline



