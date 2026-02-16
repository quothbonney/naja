#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "conditioning/eflux.h"

static std::string write_tmp(const std::string& name, const std::string& contents) {
    std::string path = std::string("/tmp/") + name;
    std::ofstream f(path);
    if (!f.is_open()) std::abort();
    f << contents;
    f.close();
    return path;
}

int main() {
    // Use real base model; just score the first reaction so we exercise the end-to-end path.
    const std::string base = "/storage/jdcarson/models/iJO1366";
    std::ifstream rxn(base + "/gem/reaction_ids.txt");
    if (!rxn.is_open()) std::abort();
    std::string r0;
    std::getline(rxn, r0);
    if (r0.empty()) std::abort();
    std::string scores = r0 + ",2.0\n";
    std::string scores_path = write_tmp("naja_test_scores.csv", scores);

    naja::conditioning::EfluxParams p;
    std::vector<std::string> cmd = {"naja", "condition", "eflux"};
    naja::conditioning::eflux_condition(base, "/tmp/naja_test_eflux_out", "row", scores_path, p, cmd);
    std::remove(scores_path.c_str());
    return 0;
}




