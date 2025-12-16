#pragma once

#include <string>
#include <vector>

namespace naja::conditioning {

struct EfluxParams {
    double Bref = 1000.0;
    double Eref_quantile = 0.95;
    double min_bound = 1e-3;
    bool shrink_only = true;
};

struct VanrijsewijkParams {
    std::string expr_operon_long_csv;
    std::string ijo_gene_reference_csv;
    std::string ijo_reaction_reference_csv;
    double missing_value = 1.0;
    std::string mode = "fold_change"; // fold_change | log2fc_pos
    bool skip_boundary = true;
};

// Applies E-Flux bound tightening from a reaction->score CSV in base reaction space.
// Writes:
//   <out_model_dir>/gem/reaction_ids.txt   (copied from base)
//   <out_model_dir>/gem/l_bounds.csv
//   <out_model_dir>/gem/u_bounds.csv
//   <out_model_dir>/gem/conditioning.json
//
// reaction_scores_csv format: lines "reaction_id,value" (no header).
void eflux_condition(const std::string& base_model_dir,
                     const std::string& out_model_dir,
                     const std::string& row_id,
                     const std::string& reaction_scores_csv,
                     const EfluxParams& p,
                     const std::vector<std::string>& command_line_tokens);

// Applies E-Flux using the Van Rijsewijk TF-KD expression dataset (operon log2FC) and an
// iJO1366 reaction reference CSV containing GPR strings.
// row_id is expected to identify (tf_gene, condition); in this workflow we use "<tf>_<condition>".
void eflux_condition_vanrijsewijk(const std::string& base_model_dir,
                                 const std::string& out_model_dir,
                                 const std::string& row_id,
                                 const VanrijsewijkParams& vr,
                                 const EfluxParams& p,
                                 const std::vector<std::string>& command_line_tokens);

} // namespace naja::conditioning


