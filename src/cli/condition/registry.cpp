#include "cli/condition/registry.h"

#include <array>
#include <string>
#include <vector>

#include "cli/condition/eflux.h"
#include "cli/condition/eflux2.h"

namespace naja::condition {
namespace {

static const std::array<std::pair<const char*, ConditionCmdFn>, 2>& table() {
    static const std::array<std::pair<const char*, ConditionCmdFn>, 2> t = {{
        {"eflux", &cmd_eflux},
        {"eflux2", &cmd_eflux2},
    }};
    return t;
}

} // namespace

ConditionCmdFn lookup_condition_cmd(const std::string& subcommand) {
    for (const auto& kv : table()) {
        if (kv.first == subcommand) return kv.second;
    }
    return nullptr;
}

std::vector<std::string> list_condition_subcommands() {
    std::vector<std::string> out;
    out.reserve(table().size());
    for (const auto& kv : table()) out.push_back(kv.first);
    return out;
}

} // namespace naja::condition






