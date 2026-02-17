#include "rounding/schedule_io.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

int main() {
    const std::string path = "/tmp/naja_test_pair_schedule.csv";
    {
        std::ofstream f(path);
        if (!f.is_open()) {
            std::cerr << "cannot open temp file\n";
            return 1;
        }
        f << "0,1,1,0\n";
        f << "2,3,0.6,0.8\n";
    }

    auto s = naja::rounding::load_pair_schedule_csv(path);
    if (s.i.size() != 2 || s.j.size() != 2 || s.c.size() != 2 || s.s.size() != 2) {
        std::cerr << "bad sizes\n";
        return 1;
    }
    if (s.i[0] != 0 || s.j[0] != 1) return 1;
    if (s.i[1] != 2 || s.j[1] != 3) return 1;

    const double n2 = s.c[1] * s.c[1] + s.s[1] * s.s[1];
    if (!(std::abs(n2 - 1.0) < 1e-12)) {
        std::cerr << "expected unit norm\n";
        return 1;
    }

    std::cout << "ok\n";
    return 0;
}



