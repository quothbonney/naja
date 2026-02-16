#include "conditioning/gpr.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace naja::conditioning::gpr {
namespace {

struct Tok {
    enum Kind { Gene, And, Or, LParen, RParen, End } kind;
    std::string text;
};

static std::string to_lower(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

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

    explicit Parser(std::string gpr) : lex(std::move(gpr)), cur(lex.next()) {}

    void eat(Tok::Kind k) {
        if (cur.kind != k) throw std::runtime_error("GPR parse error");
        cur = lex.next();
    }

    Node parse_expr() { // OR
        Node left = parse_term();
        if (cur.kind != Tok::Or) return left;
        Node out;
        out.kind = Node::Kind::Or;
        out.children.push_back(std::move(left));
        while (cur.kind == Tok::Or) {
            eat(Tok::Or);
            out.children.push_back(parse_term());
        }
        return out;
    }

    Node parse_term() { // AND
        Node left = parse_factor();
        if (cur.kind != Tok::And) return left;
        Node out;
        out.kind = Node::Kind::And;
        out.children.push_back(std::move(left));
        while (cur.kind == Tok::And) {
            eat(Tok::And);
            out.children.push_back(parse_factor());
        }
        return out;
    }

    Node parse_factor() {
        if (cur.kind == Tok::Gene) {
            Node out;
            out.kind = Node::Kind::Gene;
            out.gene = cur.text;
            eat(Tok::Gene);
            return out;
        }
        if (cur.kind == Tok::LParen) {
            eat(Tok::LParen);
            Node v = parse_expr();
            eat(Tok::RParen);
            return v;
        }
        throw std::runtime_error("GPR parse error");
    }
};

static EvalResult eval_eflux2_impl(const Node& n, const std::unordered_map<std::string, double>& gene_expr) {
    if (n.kind == Node::Kind::Gene) {
        auto it = gene_expr.find(n.gene);
        if (it == gene_expr.end()) return {0.0, false};
        double v = it->second;
        if (!std::isfinite(v) || v < 0.0) {
            throw std::runtime_error("non-finite or negative gene expression for gene: " + n.gene);
        }
        return {v, true};
    }
    if (n.children.empty()) {
        throw std::runtime_error("GPR eval error: empty operator node");
    }
    if (n.kind == Node::Kind::And) {
        bool ok = true;
        double best = 0.0;
        for (size_t i = 0; i < n.children.size(); ++i) {
            EvalResult r = eval_eflux2_impl(n.children[i], gene_expr);
            ok = ok && r.complete;
            if (i == 0) best = r.value;
            else best = std::min(best, r.value);
        }
        return {best, ok};
    }
    if (n.kind == Node::Kind::Or) {
        // E-Flux2 semantics:
        // - OR adds capacities across alternatives, but only for alternatives we can actually evaluate.
        // - Missing genes in one branch must not poison the whole OR if another branch is present.
        bool any_ok = false;
        double sum = 0.0;
        for (const auto& ch : n.children) {
            EvalResult r = eval_eflux2_impl(ch, gene_expr);
            if (r.complete) {
                any_ok = true;
                sum += r.value;
            }
        }
        return {sum, any_ok};
    }
    throw std::runtime_error("GPR eval error");
}

} // namespace

Node parse(const std::string& gpr) {
    Parser p(gpr);
    Node n = p.parse_expr();
    if (p.cur.kind != Tok::End) throw std::runtime_error("GPR parse trailing tokens");
    return n;
}

EvalResult eval_eflux2(const Node& n, const std::unordered_map<std::string, double>& gene_expr) {
    return eval_eflux2_impl(n, gene_expr);
}

} // namespace naja::conditioning::gpr



