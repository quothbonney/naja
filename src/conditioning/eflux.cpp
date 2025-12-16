#include "conditioning/eflux.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "csv_loader.h"
#include "utils.h"

namespace naja::conditioning {
namespace {

struct CsvRow {
    std::vector<std::string> fields;
};

CsvRow parse_csv_line(const std::string& line) {
    CsvRow r;
    r.fields.reserve(8);
    std::string cur;
    bool in_quotes = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (in_quotes) {
            if (c == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    cur.push_back('"');
                    ++i;
                } else {
                    in_quotes = false;
                }
            } else {
                cur.push_back(c);
            }
        } else {
            if (c == '"') {
                in_quotes = true;
            } else if (c == ',') {
                r.fields.push_back(cur);
                cur.clear();
            } else {
                cur.push_back(c);
            }
        }
    }
    r.fields.push_back(cur);
    return r;
}

std::string to_lower(std::string s) {
    for (char& c : s) {
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
    }
    return s;
}

std::string norm_gene_symbol(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == ' ' || c == '_') continue;
        if (c >= 'A' && c <= 'Z') out.push_back((char)(c - 'A' + 'a'));
        else out.push_back(c);
    }
    // trim not needed; source CSV already clean-ish
    return out;
}

struct Tok {
    enum Kind { Gene, And, Or, LParen, RParen, End } kind;
    std::string text;
};

struct Lexer {
    std::string s;
    size_t i = 0;
    explicit Lexer(std::string in) : s(std::move(in)) {}
    static bool is_ws(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }
    static bool is_id(char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '.';
    }
    Tok next() {
        while (i < s.size() && is_ws(s[i])) ++i;
        if (i >= s.size()) return {Tok::End, ""};
        char c = s[i];
        if (c == '(') {
            ++i;
            return {Tok::LParen, "("};
        }
        if (c == ')') {
            ++i;
            return {Tok::RParen, ")"};
        }
        if (is_id(c)) {
            size_t j = i;
            while (j < s.size() && is_id(s[j])) ++j;
            std::string w = s.substr(i, j - i);
            i = j;
            std::string wl = to_lower(w);
            if (wl == "and") return {Tok::And, w};
            if (wl == "or") return {Tok::Or, w};
            return {Tok::Gene, w};
        }
        throw std::runtime_error("invalid GPR character: " + std::string(1, c));
    }
};

struct Parser {
    Lexer lex;
    Tok cur;
    const std::unordered_map<std::string, double>& gene_expr;
    double missing_value;

    Parser(std::string gpr,
           const std::unordered_map<std::string, double>& ge,
           double mv)
        : lex(std::move(gpr)), cur(lex.next()), gene_expr(ge), missing_value(mv) {}

    void eat(Tok::Kind k) {
        if (cur.kind != k) throw std::runtime_error("GPR parse error");
        cur = lex.next();
    }

    double gene_value(const std::string& g) const {
        auto it = gene_expr.find(g);
        if (it != gene_expr.end()) return it->second;
        return missing_value;
    }

    double parse_expr() { // OR
        double v = parse_term();
        while (cur.kind == Tok::Or) {
            eat(Tok::Or);
            double r = parse_term();
            v = std::max(v, r);
        }
        return v;
    }

    double parse_term() { // AND
        double v = parse_factor();
        while (cur.kind == Tok::And) {
            eat(Tok::And);
            double r = parse_factor();
            v = std::min(v, r);
        }
        return v;
    }

    double parse_factor() {
        if (cur.kind == Tok::Gene) {
            std::string g = cur.text;
            eat(Tok::Gene);
            return gene_value(g);
        }
        if (cur.kind == Tok::LParen) {
            eat(Tok::LParen);
            double v = parse_expr();
            eat(Tok::RParen);
            return v;
        }
        throw std::runtime_error("GPR parse error");
    }
};

std::vector<std::string> read_lines_nonempty(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("cannot read: " + path);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(f, line)) {
        // trim
        auto is_ws = [](unsigned char ch) { return ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n'; };
        while (!line.empty() && is_ws((unsigned char)line.front())) line.erase(line.begin());
        while (!line.empty() && is_ws((unsigned char)line.back())) line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        lines.push_back(line);
    }
    if (lines.empty()) throw std::runtime_error("empty file: " + path);
    return lines;
}

std::unordered_map<std::string, double> read_scores_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("cannot read: " + path);
    std::unordered_map<std::string, double> m;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto comma = line.find(',');
        if (comma == std::string::npos) {
            throw std::runtime_error("invalid scores csv line (expected reaction,value): " + line);
        }
        std::string rid = line.substr(0, comma);
        std::string vs = line.substr(comma + 1);
        // trim
        auto trim = [](std::string& s) {
            auto is_ws = [](unsigned char ch) { return ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n'; };
            while (!s.empty() && is_ws((unsigned char)s.front())) s.erase(s.begin());
            while (!s.empty() && is_ws((unsigned char)s.back())) s.pop_back();
        };
        trim(rid);
        trim(vs);
        if (rid.empty() || vs.empty()) continue;
        double v = std::stod(vs);
        m[rid] = v;
    }
    if (m.empty()) throw std::runtime_error("no scores in: " + path);
    return m;
}

std::unordered_map<std::string, double> build_gene_expr_from_vanrijsewijk(
    const std::string& expr_operon_long_csv,
    const std::string& ijo_gene_reference_csv,
    const std::string& tf_gene,
    const std::string& condition,
    double missing_value,
    const std::string& mode) {
    // operon -> log2fc
    std::unordered_map<std::string, double> operon_log2fc;
    {
        std::ifstream f(expr_operon_long_csv);
        if (!f.is_open()) throw std::runtime_error("cannot read: " + expr_operon_long_csv);
        std::string line;
        if (!std::getline(f, line)) throw std::runtime_error("empty: " + expr_operon_long_csv);
        auto hdr = parse_csv_line(line).fields;
        if (hdr.size() < 6) throw std::runtime_error("unexpected header in expr_operon_long");
        int tf_i = -1, op_i = -1, cond_i = -1, log2_i = -1;
        for (int i = 0; i < (int)hdr.size(); ++i) {
            std::string h = hdr[i];
            if (h == "tf_gene") tf_i = i;
            else if (h == "operon") op_i = i;
            else if (h == "condition") cond_i = i;
            else if (h == "log2fc") log2_i = i;
        }
        if (tf_i < 0 || op_i < 0 || cond_i < 0 || log2_i < 0) {
            throw std::runtime_error("expr_operon_long missing required columns");
        }
        std::string tf_l = to_lower(tf_gene);
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            auto row = parse_csv_line(line).fields;
            if ((int)row.size() <= std::max(std::max(tf_i, op_i), std::max(cond_i, log2_i))) continue;
            if (to_lower(row[tf_i]) != tf_l) continue;
            if (row[cond_i] != condition) continue;
            std::string oper = row[op_i];
            std::string v = row[log2_i];
            if (v.empty()) continue;
            operon_log2fc[norm_gene_symbol(oper)] = std::stod(v);
        }
        if (operon_log2fc.empty()) {
            throw std::runtime_error("no expression rows for tf_gene=" + tf_gene + " condition=" + condition);
        }
    }

    // iJO gene id -> expr
    std::unordered_map<std::string, double> gene_expr;
    {
        std::ifstream f(ijo_gene_reference_csv);
        if (!f.is_open()) throw std::runtime_error("cannot read: " + ijo_gene_reference_csv);
        std::string line;
        if (!std::getline(f, line)) throw std::runtime_error("empty: " + ijo_gene_reference_csv);
        auto hdr = parse_csv_line(line).fields;
        int id_i = -1, norm_i = -1;
        for (int i = 0; i < (int)hdr.size(); ++i) {
            if (hdr[i] == "iJO_gene_id") id_i = i;
            else if (hdr[i] == "iJO_gene_name_norm") norm_i = i;
        }
        if (id_i < 0 || norm_i < 0) throw std::runtime_error("iJO gene reference missing columns");
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            auto row = parse_csv_line(line).fields;
            if ((int)row.size() <= std::max(id_i, norm_i)) continue;
            std::string gid = row[id_i];
            std::string gnorm = row[norm_i];
            double log2fc = missing_value; // placeholder: will overwrite if found
            auto it = operon_log2fc.find(gnorm);
            if (it == operon_log2fc.end()) {
                gene_expr[gid] = missing_value;
                continue;
            }
            double l2 = it->second;
            if (mode == "fold_change") gene_expr[gid] = std::pow(2.0, l2);
            else if (mode == "log2fc_pos") gene_expr[gid] = std::max(l2, 0.0);
            else throw std::runtime_error("unknown mode: " + mode);
        }
        if (gene_expr.empty()) throw std::runtime_error("gene reference produced empty gene_expr");
    }
    return gene_expr;
}

std::unordered_map<std::string, double> build_reaction_scores_from_gpr(
    const std::string& ijo_reaction_reference_csv,
    const std::unordered_map<std::string, double>& gene_expr,
    double missing_value,
    bool skip_boundary) {
    std::unordered_map<std::string, double> rxn_scores;
    std::ifstream f(ijo_reaction_reference_csv);
    if (!f.is_open()) throw std::runtime_error("cannot read: " + ijo_reaction_reference_csv);
    std::string line;
    if (!std::getline(f, line)) throw std::runtime_error("empty: " + ijo_reaction_reference_csv);
    auto hdr = parse_csv_line(line).fields;
    int id_i = -1, gpr_i = -1;
    for (int i = 0; i < (int)hdr.size(); ++i) {
        if (hdr[i] == "ID") id_i = i;
        else if (hdr[i] == "GPR") gpr_i = i;
    }
    if (id_i < 0 || gpr_i < 0) throw std::runtime_error("reaction reference missing ID/GPR columns");
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto row = parse_csv_line(line).fields;
        if ((int)row.size() <= std::max(id_i, gpr_i)) continue;
        std::string rid = row[id_i];
        std::string gpr = row[gpr_i];
        if (gpr.empty()) continue;
        if (skip_boundary) {
            if (rid.rfind("EX_", 0) == 0) continue;
            if (rid.rfind("DM_", 0) == 0) continue;
        }
        Parser p(gpr, gene_expr, missing_value);
        double v = p.parse_expr();
        if (p.cur.kind != Tok::End) throw std::runtime_error("GPR parse trailing tokens for " + rid);
        rxn_scores[rid] = v;
    }
    if (rxn_scores.empty()) throw std::runtime_error("no reaction scores computed (GPR parse?)");
    return rxn_scores;
}

double quantile_inplace(std::vector<double>& v, double q) {
    if (v.empty()) throw std::runtime_error("empty vector for quantile");
    if (q < 0.0 || q > 1.0) throw std::runtime_error("quantile out of range");
    size_t k = (size_t)std::floor(q * (double)(v.size() - 1));
    std::nth_element(v.begin(), v.begin() + (ptrdiff_t)k, v.end());
    return v[k];
}

void write_vector_csv(const std::string& path, const Eigen::VectorXd& v) {
    std::ofstream f(path);
    if (!f.is_open()) throw std::runtime_error("cannot write: " + path);
    for (int i = 0; i < v.size(); ++i) {
        f << std::setprecision(12) << v[i] << "\n";
    }
}

void write_reaction_ids(const std::string& path, const std::vector<std::string>& ids) {
    std::ofstream f(path);
    if (!f.is_open()) throw std::runtime_error("cannot write: " + path);
    for (const auto& id : ids) f << id << "\n";
}

void json_string(std::ostream& o, const std::string& s) {
    o << "\"";
    for (char c : s) {
        switch (c) {
            case '\\': o << "\\\\"; break;
            case '"': o << "\\\""; break;
            case '\n': o << "\\n"; break;
            case '\r': o << "\\r"; break;
            case '\t': o << "\\t"; break;
            default: o << c; break;
        }
    }
    o << "\"";
}

} // namespace

void eflux_condition(const std::string& base_model_dir,
                     const std::string& out_model_dir,
                     const std::string& row_id,
                     const std::string& reaction_scores_csv,
                     const EfluxParams& p,
                     const std::vector<std::string>& command_line_tokens) {
    const std::string base_gem = base_model_dir + "/gem";
    const std::string out_gem = out_model_dir + "/gem";
    ensure_dir(out_gem);

    const std::string rxn_ids_path = base_gem + "/reaction_ids.txt";
    const std::string lb_path = base_gem + "/l_bounds.csv";
    const std::string ub_path = base_gem + "/u_bounds.csv";

    std::vector<std::string> rxn_ids = read_lines_nonempty(rxn_ids_path);
    Eigen::VectorXd lb0 = csv::loadVector(lb_path);
    Eigen::VectorXd ub0 = csv::loadVector(ub_path);
    if (lb0.size() != (int)rxn_ids.size() || ub0.size() != (int)rxn_ids.size()) {
        throw std::runtime_error("base bounds length mismatch vs reaction_ids");
    }

    auto scores = read_scores_csv(reaction_scores_csv);

    // Unknown reaction IDs are an error.
    {
        std::unordered_map<std::string, int> present;
        present.reserve(rxn_ids.size());
        for (const auto& id : rxn_ids) present[id] = 1;
        for (const auto& kv : scores) {
            if (present.find(kv.first) == present.end()) {
                throw std::runtime_error("unknown reaction_id in scores: " + kv.first);
            }
        }
    }

    std::vector<double> pos;
    pos.reserve(scores.size());
    for (const auto& kv : scores) {
        double v = kv.second;
        if (std::isfinite(v) && v > 0.0) pos.push_back(v);
    }
    if (pos.empty()) throw std::runtime_error("all scores are <=0 or non-finite");
    double Eref = quantile_inplace(pos, p.Eref_quantile);
    if (Eref <= 0.0) throw std::runtime_error("Eref <= 0");

    Eigen::VectorXd lb = lb0;
    Eigen::VectorXd ub = ub0;
    int tightened = 0;

    for (int i = 0; i < (int)rxn_ids.size(); ++i) {
        const std::string& rid = rxn_ids[i];
        auto it = scores.find(rid);
        if (it == scores.end()) continue;
        double Ej = it->second;
        if (!std::isfinite(Ej)) continue;

        double cap = p.Bref * (Ej / Eref);
        if (cap < p.min_bound) cap = p.min_bound;

        double lb1 = lb0[i];
        double ub1 = ub0[i];
        double new_lb = lb1;
        double new_ub = ub1;

        if (lb1 < 0.0 && ub1 > 0.0) {
            new_lb = -cap;
            new_ub = cap;
        } else if (ub1 > 0.0 && lb1 >= 0.0) {
            new_lb = lb1;
            new_ub = cap;
        } else if (lb1 < 0.0 && ub1 <= 0.0) {
            new_lb = -cap;
            new_ub = ub1;
        } else {
            continue;
        }

        if (p.shrink_only) {
            if (new_lb < lb1) new_lb = lb1;
            if (new_ub > ub1) new_ub = ub1;
        }

        if (new_lb > lb1 || new_ub < ub1) tightened++;
        if (new_lb > new_ub) {
            throw std::runtime_error("infeasible bounds for " + rid);
        }

        lb[i] = new_lb;
        ub[i] = new_ub;
    }

    write_reaction_ids(out_gem + "/reaction_ids.txt", rxn_ids);
    write_vector_csv(out_gem + "/l_bounds.csv", lb);
    write_vector_csv(out_gem + "/u_bounds.csv", ub);

    // conditioning.json
    {
        std::ofstream f(out_gem + "/conditioning.json");
        if (!f.is_open()) throw std::runtime_error("cannot write conditioning.json");
        f << "{\n";
        f << "  \"method\": \"eflux\",\n";
        f << "  \"base_model\": {\"path\": "; json_string(f, make_absolute_path(base_model_dir)); f << "},\n";
        f << "  \"row_id\": "; json_string(f, row_id); f << ",\n";
        f << "  \"gene_id_namespace\": \"reaction_id\",\n";
        f << "  \"input_expression\": {\"type\": \"csv\", \"path\": "; json_string(f, make_absolute_path(reaction_scores_csv)); f << "},\n";
        f << "  \"parameters\": {\n";
        f << "    \"Bref\": " << std::setprecision(12) << p.Bref << ",\n";
        f << "    \"Eref_quantile\": " << std::setprecision(12) << p.Eref_quantile << ",\n";
        f << "    \"min_bound\": " << std::setprecision(12) << p.min_bound << ",\n";
        f << "    \"shrink_only\": " << (p.shrink_only ? "true" : "false") << ",\n";
        f << "    \"missing_reaction_policy\": \"unchanged\"\n";
        f << "  },\n";
        f << "  \"created_at\": "; json_string(f, current_timestamp()); f << ",\n";
        f << "  \"command_line\": [";
        for (size_t i = 0; i < command_line_tokens.size(); ++i) {
            if (i) f << ", ";
            json_string(f, command_line_tokens[i]);
        }
        f << "],\n";
        f << "  \"stats\": {\"tightened\": " << tightened << ", \"Eref\": " << std::setprecision(12) << Eref << "}\n";
        f << "}\n";
    }
}

static std::pair<std::string, std::string> parse_row_id_tf_cond(const std::string& row_id) {
    std::string s = row_id;
    const std::string prefix = "iJO1366_EFLUX_";
    if (s.rfind(prefix, 0) == 0) s = s.substr(prefix.size());
    auto pos = s.find_last_of('_');
    if (pos == std::string::npos) {
        throw std::runtime_error("row-id must be <tf>_<condition>: " + row_id);
    }
    std::string tf = s.substr(0, pos);
    std::string cond = s.substr(pos + 1);
    if (cond != "LB" && cond != "glucose" && cond != "glycerol") {
        throw std::runtime_error("row-id condition must be LB|glucose|glycerol: " + row_id);
    }
    if (tf.empty()) throw std::runtime_error("empty tf in row-id: " + row_id);
    return {tf, cond};
}

void eflux_condition_vanrijsewijk(const std::string& base_model_dir,
                                 const std::string& out_model_dir,
                                 const std::string& row_id,
                                 const VanrijsewijkParams& vr,
                                 const EfluxParams& p,
                                 const std::vector<std::string>& command_line_tokens) {
    auto [tf, cond] = parse_row_id_tf_cond(row_id);
    auto gene_expr = build_gene_expr_from_vanrijsewijk(
        vr.expr_operon_long_csv,
        vr.ijo_gene_reference_csv,
        tf,
        cond,
        vr.missing_value,
        vr.mode);
    auto rxn_scores = build_reaction_scores_from_gpr(
        vr.ijo_reaction_reference_csv,
        gene_expr,
        vr.missing_value,
        vr.skip_boundary);

    // Restrict to base reaction_ids; the base gem is the canonical reaction space.
    std::unordered_map<std::string, int> base_present;
    {
        std::vector<std::string> base_rxn_ids = read_lines_nonempty(base_model_dir + "/gem/reaction_ids.txt");
        base_present.reserve(base_rxn_ids.size());
        for (const auto& r : base_rxn_ids) base_present[r] = 1;
    }

    // Write a temporary reaction-scores CSV in out_model_dir for auditability.
    const std::string out_gem = out_model_dir + "/gem";
    ensure_dir(out_gem);
    const std::string scores_path = out_gem + "/reaction_scores.csv";
    {
        std::ofstream f(scores_path);
        if (!f.is_open()) throw std::runtime_error("cannot write: " + scores_path);
        int n = 0;
        for (const auto& kv : rxn_scores) {
            if (base_present.find(kv.first) == base_present.end()) continue;
            f << kv.first << "," << std::setprecision(12) << kv.second << "\n";
            n++;
        }
        if (n == 0) {
            throw std::runtime_error("no reaction scores in base reaction space after filtering");
        }
    }

    eflux_condition(base_model_dir, out_model_dir, row_id, scores_path, p, command_line_tokens);

    // Patch conditioning.json to point at vanrijsewijk inputs, not just the derived CSV.
    {
        const std::string cj = out_gem + "/conditioning.json";
        std::ofstream f(cj);
        if (!f.is_open()) throw std::runtime_error("cannot write conditioning.json");
        f << "{\n";
        f << "  \"method\": \"eflux\",\n";
        f << "  \"base_model\": {\"path\": "; json_string(f, make_absolute_path(base_model_dir)); f << "},\n";
        f << "  \"row_id\": "; json_string(f, row_id); f << ",\n";
        f << "  \"gene_id_namespace\": \"iJO1366.gene.id\",\n";
        f << "  \"input_expression\": {\n";
        f << "    \"type\": \"csv\",\n";
        f << "    \"path\": "; json_string(f, make_absolute_path(vr.expr_operon_long_csv)); f << ",\n";
        f << "    \"selector\": {\"tf_gene\": "; json_string(f, tf); f << ", \"condition\": "; json_string(f, cond); f << ", \"mode\": "; json_string(f, vr.mode); f << "}\n";
        f << "  },\n";
        f << "  \"parameters\": {\n";
        f << "    \"Bref\": " << std::setprecision(12) << p.Bref << ",\n";
        f << "    \"Eref_quantile\": " << std::setprecision(12) << p.Eref_quantile << ",\n";
        f << "    \"min_bound\": " << std::setprecision(12) << p.min_bound << ",\n";
        f << "    \"missing_value\": " << std::setprecision(12) << vr.missing_value << ",\n";
        f << "    \"skip_boundary\": " << (vr.skip_boundary ? "true" : "false") << ",\n";
        f << "    \"shrink_only\": " << (p.shrink_only ? "true" : "false") << "\n";
        f << "  },\n";
        f << "  \"created_at\": "; json_string(f, current_timestamp()); f << ",\n";
        f << "  \"command_line\": [";
        for (size_t i = 0; i < command_line_tokens.size(); ++i) {
            if (i) f << ", ";
            json_string(f, command_line_tokens[i]);
        }
        f << "],\n";
        f << "  \"derived\": {\"reaction_scores_csv\": "; json_string(f, make_absolute_path(scores_path)); f << "}\n";
        f << "}\n";
    }
}

} // namespace naja::conditioning


