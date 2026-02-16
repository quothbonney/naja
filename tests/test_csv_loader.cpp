#include <cassert>
#include <cstdio>
#include <fstream>
#include <string>

#include "csv_loader.h"

static std::string write_tmp(const std::string& name, const std::string& contents) {
    std::string path = std::string("/tmp/") + name;
    std::ofstream f(path);
    assert(f.is_open());
    f << contents;
    f.close();
    return path;
}

int main() {
    std::string p = write_tmp("naja_test_vec.csv", "1.0\n2.5\n3.0\n");
    Eigen::VectorXd v = csv::loadVector(p);
    assert(v.size() == 3);
    assert(v[0] == 1.0);
    assert(v[1] == 2.5);
    assert(v[2] == 3.0);
    std::remove(p.c_str());
    return 0;
}




