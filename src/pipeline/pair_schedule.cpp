#include "pipeline/pair_schedule.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace naja::pipeline {

PairSchedule load_pair_schedule_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("cannot open pair schedule: " + path);
    }

    std::vector<int> vi;
    std::vector<int> vj;
    std::vector<double> vc;
    std::vector<double> vs;

    std::string line;
    int lineno = 0;
    while (std::getline(f, line)) {
        ++lineno;
        // trim leading ws
        while (!line.empty() && (line.front() == ' ' || line.front() == '\t')) line.erase(line.begin());
        if (line.empty() || line[0] == '#') continue;

        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> cells;
        while (std::getline(ss, cell, ',')) {
            cells.push_back(cell);
        }
        if (cells.size() != 4) {
            throw std::runtime_error("pair schedule parse error at line " + std::to_string(lineno) + ": expected 4 columns");
        }
        int ii = std::stoi(cells[0]);
        int jj = std::stoi(cells[1]);
        double c = std::stod(cells[2]);
        double s = std::stod(cells[3]);
        if (ii < 0 || jj < 0) {
            throw std::runtime_error("pair schedule parse error at line " + std::to_string(lineno) + ": negative index");
        }
        if (!std::isfinite(c) || !std::isfinite(s)) {
            throw std::runtime_error("pair schedule parse error at line " + std::to_string(lineno) + ": non-finite c/s");
        }
        // sanity: should be unit-norm (allow slack)
        const double n2 = c * c + s * s;
        if (!(n2 > 0.0)) {
            throw std::runtime_error("pair schedule parse error at line " + std::to_string(lineno) + ": zero norm");
        }
        vi.push_back(ii);
        vj.push_back(jj);
        vc.push_back(c);
        vs.push_back(s);
    }

    if (vi.empty()) {
        throw std::runtime_error("pair schedule is empty: " + path);
    }

    PairSchedule out;
    out.i.resize((int)vi.size());
    out.j.resize((int)vj.size());
    out.c.resize((int)vc.size());
    out.s.resize((int)vs.size());
    for (int k = 0; k < (int)vi.size(); ++k) {
        out.i[k] = vi[k];
        out.j[k] = vj[k];
        out.c[k] = vc[k];
        out.s[k] = vs[k];
    }
    return out;
}

void write_pair_schedule_csv(const std::string& path, const PairSchedule& sched) {
    if (sched.i.size() != sched.j.size() || sched.i.size() != sched.c.size() || sched.i.size() != sched.s.size()) {
        throw std::invalid_argument("write_pair_schedule_csv: schedule arrays must have same length");
    }
    std::ofstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("cannot write pair schedule: " + path);
    }
    for (int k = 0; k < sched.i.size(); ++k) {
        f << sched.i[k] << "," << sched.j[k] << "," << sched.c[k] << "," << sched.s[k] << "\n";
    }
}

} // namespace naja::pipeline



