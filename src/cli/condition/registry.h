#pragma once

#include <string>
#include <vector>

namespace naja::condition {

using ConditionCmdFn = int (*)(int argc, char** argv);

ConditionCmdFn lookup_condition_cmd(const std::string& subcommand);
std::vector<std::string> list_condition_subcommands();

} // namespace naja::condition






