#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "conditioning/eflux2.h"
#include "utils.h"

static void write_text(const std::string& path, const std::string& s) {
    std::ofstream f(path);
    if (!f.is_open()) std::abort();
    f << s;
}

static std::vector<double> read_lines_as_double(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) std::abort();
    std::vector<double> out;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        out.push_back(std::stod(line));
    }
    return out;
}

static void require(bool ok) {
    if (!ok) std::abort();
}

int main() {
    // Build a tiny fake base model in /tmp.
    const std::string base = "/tmp/naja_test_eflux2_base";
    const std::string out = "/tmp/naja_test_eflux2_out";
    ensure_dir(base + "/gem");
    ensure_dir(out);

    // 3 reactions: reversible, forward-only, backward-only
    write_text(base + "/gem/reaction_ids.txt", "R1\nR2\nR3\n");
    write_text(base + "/gem/l_bounds.csv", "-1000\n0\n-1000\n");
    write_text(base + "/gem/u_bounds.csv", "1000\n1000\n0\n");

    // Provide scores for R1 and R2 only; R3 should be unchanged.
    const std::string scores = "/tmp/naja_test_eflux2_scores.csv";
    write_text(scores, "R1,5\nR2,10\n");

    naja::conditioning::Eflux2Params p;
    p.Bref = 500.0;
    p.Eref_quantile = 0.5; // deterministic for [5,10] => Eref=5
    p.min_bound = 1e-6;
    p.shrink_only = true;
    std::vector<std::string> cmd = {"naja", "condition", "eflux2"};

    naja::conditioning::eflux2_condition(base, out, "row", scores, p, cmd);

    auto lb = read_lines_as_double(out + "/gem/l_bounds.csv");
    auto ub = read_lines_as_double(out + "/gem/u_bounds.csv");
    require(lb.size() == 3 && ub.size() == 3);

    // R1 cap = 500 => [-500, 500]
    require(lb[0] == -500.0);
    require(ub[0] == 500.0);

    // R2 cap = 1000 => ub stays 1000 (equal), lb stays 0
    require(lb[1] == 0.0);
    require(ub[1] == 1000.0);

    // R3 missing score => unchanged
    require(lb[2] == -1000.0);
    require(ub[2] == 0.0);

    return 0;
}






