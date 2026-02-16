#include "cli/condition/eflux2.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "conditioning/eflux2.h"
#include "cli/sample/common.h"

namespace {

[[noreturn]] void die_usage(const std::string& msg) {
    std::cerr << "error: " << msg << "\n\n";
    std::cerr << "usage:\n";
    std::cerr << "  naja condition eflux2 \\\n";
    std::cerr << "    --base-model-dir <dir> \\\n";
    std::cerr << "    --out-model-dir <dir> \\\n";
    std::cerr << "    --row-id <string> \\\n";
    std::cerr << "    (--reaction-scores <csv> | --vanrijsewijk-expr-long <csv> --ijo-gene-ref <csv> --ijo-reaction-ref <csv>)\n";
    std::exit(2);
}

} // namespace

namespace naja::condition {

int cmd_eflux2(int argc, char** argv) {
    std::string base_model_dir;
    std::string out_model_dir;
    std::string row_id;
    std::string reaction_scores;
    naja::conditioning::Eflux2Params p;
    naja::conditioning::VanrijsewijkParams2 vr;
    bool use_vr = false;

    std::vector<std::string> cmdline;
    cmdline.reserve((size_t)argc + 3);
    cmdline.push_back("naja");
    cmdline.push_back("condition");
    cmdline.push_back("eflux2");
    for (int i = 0; i < argc; ++i) cmdline.push_back(argv[i]);

    for (int i = 0; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--base-model-dir") base_model_dir = naja::cli::sample::next_arg(i, argc, argv, a);
        else if (a == "--out-model-dir") out_model_dir = naja::cli::sample::next_arg(i, argc, argv, a);
        else if (a == "--row-id") row_id = naja::cli::sample::next_arg(i, argc, argv, a);
        else if (a == "--reaction-scores") reaction_scores = naja::cli::sample::next_arg(i, argc, argv, a);

        else if (a == "--vanrijsewijk-expr-long") { vr.expr_operon_long_csv = naja::cli::sample::next_arg(i, argc, argv, a); use_vr = true; }
        else if (a == "--ijo-gene-ref") { vr.ijo_gene_reference_csv = naja::cli::sample::next_arg(i, argc, argv, a); use_vr = true; }
        else if (a == "--ijo-reaction-ref") { vr.ijo_reaction_reference_csv = naja::cli::sample::next_arg(i, argc, argv, a); use_vr = true; }
        else if (a == "--mode") { vr.mode = naja::cli::sample::next_arg(i, argc, argv, a); use_vr = true; }
        else if (a == "--no-skip-boundary") { vr.skip_boundary = false; use_vr = true; }

        else if (a == "--Bref") p.Bref = std::stod(naja::cli::sample::next_arg(i, argc, argv, a));
        else if (a == "--Eref-q") p.Eref_quantile = std::stod(naja::cli::sample::next_arg(i, argc, argv, a));
        else if (a == "--min-bound") p.min_bound = std::stod(naja::cli::sample::next_arg(i, argc, argv, a));
        else if (a == "--no-shrink-only") p.shrink_only = false;
        else if (a == "--help" || a == "-h") die_usage("");
        else die_usage("unknown flag: " + a);
    }

    if (base_model_dir.empty()) die_usage("missing --base-model-dir");
    if (out_model_dir.empty()) die_usage("missing --out-model-dir");
    if (row_id.empty()) die_usage("missing --row-id");
    if (!use_vr && reaction_scores.empty()) die_usage("missing --reaction-scores (or vanrijsewijk flags)");
    if (use_vr && (!reaction_scores.empty())) die_usage("cannot pass both --reaction-scores and vanrijsewijk flags");
    if (use_vr) {
        if (vr.expr_operon_long_csv.empty()) die_usage("missing --vanrijsewijk-expr-long");
        if (vr.ijo_gene_reference_csv.empty()) die_usage("missing --ijo-gene-ref");
        if (vr.ijo_reaction_reference_csv.empty()) die_usage("missing --ijo-reaction-ref");
    }

    if (use_vr) {
        naja::conditioning::eflux2_condition_vanrijsewijk(base_model_dir, out_model_dir, row_id, vr, p, cmdline);
    } else {
        naja::conditioning::eflux2_condition(base_model_dir, out_model_dir, row_id, reaction_scores, p, cmdline);
    }
    return 0;
}

} // namespace naja::condition






