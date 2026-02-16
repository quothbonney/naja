#include <cstdlib>
#include <string>
#include <unordered_map>

#include "conditioning/gpr.h"

static void require(bool ok, const std::string& msg) {
    if (!ok) {
        (void)msg;
        std::abort();
    }
}

int main() {
    // E-Flux2 semantics:
    // AND => min, OR => sum, and missing genes must mark the expression as incomplete.
    //
    // (g1 AND g2) OR g3  with g1=2, g2=5, g3=7:
    //   AND part = min(2,5)=2
    //   OR sum   = 2 + 7 = 9
    const std::string gpr_s = "(g1 and g2) or g3";
    auto ast = naja::conditioning::gpr::parse(gpr_s);

    std::unordered_map<std::string, double> E = {{"g1", 2.0}, {"g2", 5.0}, {"g3", 7.0}};
    auto r = naja::conditioning::gpr::eval_eflux2(ast, E);
    require(r.complete, "expected complete");
    require(r.value == 9.0, "expected OR=sum and AND=min");

    // Missing gene in one OR branch must not poison the whole OR if another branch is present.
    // Here, removing g2 makes (g1 AND g2) incomplete, but g3 is still present, so:
    //   OR=sum over present branches => 7
    E.erase("g2");
    auto r2 = naja::conditioning::gpr::eval_eflux2(ast, E);
    require(r2.complete, "expected complete because g3 branch is present");
    require(r2.value == 7.0, "expected OR to sum only present branches");

    // If all OR branches are missing/incomplete, it should be incomplete.
    E.erase("g3");
    auto r3 = naja::conditioning::gpr::eval_eflux2(ast, E);
    require(!r3.complete, "expected incomplete when no OR branch is evaluable");
    return 0;
}



