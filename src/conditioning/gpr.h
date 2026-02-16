#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace naja::conditioning::gpr {

struct Node {
    enum class Kind { Gene, And, Or } kind;
    std::string gene;              // for Gene
    std::vector<Node> children;    // for And/Or
};

// Parse a GPR string containing gene ids, parentheses, and operators AND/OR (case-insensitive).
// Throws std::runtime_error on any parse error.
Node parse(const std::string& gpr);

struct EvalResult {
    double value = 0.0;
    bool complete = true; // false if any gene referenced in the tree is missing from gene_expr
};

// Evaluate with E-Flux2 semantics:
// - AND => min(children)
// - OR  => sum(children)
// If any leaf gene is missing from gene_expr, complete=false.
EvalResult eval_eflux2(const Node& n, const std::unordered_map<std::string, double>& gene_expr);

} // namespace naja::conditioning::gpr






