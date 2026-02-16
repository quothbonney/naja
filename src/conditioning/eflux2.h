#pragma once

#include <string>
#include <vector>

namespace naja::conditioning {

struct Eflux2Params {
    // Scale expression-derived capacities into the typical GEM bound scale (same idea as existing E-Flux).
    double Bref = 1000.0;
    double Eref_quantile = 0.95;
    double min_bound = 1e-3;
    bool shrink_only = true; // do not loosen bounds vs base
};

struct VanrijsewijkParams2 {
    std::string expr_operon_long_csv;
    std::string ijo_gene_reference_csv;
    std::string ijo_reaction_reference_csv;
    std::string mode = "fold_change"; // fold_change | log2fc_pos
    bool skip_boundary = true;
};

// E-Flux2 bound conditioning from a reaction->score CSV in base reaction space.
// Semantics differ from E-Flux only in how reaction scores are expected to have been computed
// (e.g., GPR OR=sum, AND=min; missing => unchanged). This function just consumes reaction scores.
//
// Writes:
//   <out_model_dir>/gem/reaction_ids.txt   (copied from base)
//   <out_model_dir>/gem/l_bounds.csv
//   <out_model_dir>/gem/u_bounds.csv
//   <out_model_dir>/gem/conditioning.json
//
// reaction_scores_csv format: lines "reaction_id,value" (no header).
void eflux2_condition(const std::string& base_model_dir,
                      const std::string& out_model_dir,
                      const std::string& row_id,
                      const std::string& reaction_scores_csv,
                      const Eflux2Params& p,
                      const std::vector<std::string>& command_line_tokens);

// E-Flux2 using Van Rijsewijk TF-KD expression dataset.
// row_id identifies (tf_gene, condition); expected "<tf>_<condition>" (optionally prefixed).
void eflux2_condition_vanrijsewijk(const std::string& base_model_dir,
                                  const std::string& out_model_dir,
                                  const std::string& row_id,
                                  const VanrijsewijkParams2& vr,
                                  const Eflux2Params& p,
                                  const std::vector<std::string>& command_line_tokens);

} // namespace naja::conditioning






