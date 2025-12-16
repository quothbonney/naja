#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "cli/condition/eflux.h"

int main() {
    const std::string base = "/storage/jdcarson/models/iJO1366";
    std::ifstream rxn(base + "/gem/reaction_ids.txt");
    if (!rxn.is_open()) std::abort();
    std::string r0, r1;
    std::getline(rxn, r0);
    std::getline(rxn, r1);
    if (r0.empty() || r1.empty()) std::abort();

    const std::string scores_path = "/tmp/naja_test_scores2.csv";
    {
        std::ofstream f(scores_path);
        if (!f.is_open()) std::abort();
        f << r0 << ",2.0\n";
        f << r1 << ",0.5\n";
    }

    std::vector<std::string> args_s = {
        "--base-model-dir", base,
        "--out-model-dir", "/tmp/naja_test_condition_out",
        "--row-id", "row",
        "--reaction-scores", scores_path,
    };
    std::vector<char*> args;
    for (auto& s : args_s) args.push_back(s.data());
    int rc = naja::condition::cmd_eflux((int)args.size(), args.data());
    if (rc != 0) return rc;
    std::remove(scores_path.c_str());
    return 0;
}


