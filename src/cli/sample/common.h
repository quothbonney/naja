#pragma once

#include <string>

namespace naja::cli::sample {

[[noreturn]] void die_usage(const std::string& msg);
std::string next_arg(int& i, int argc, char** argv, const std::string& flag);

void require_nonempty_file(const std::string& path, const std::string& what);
bool file_is_empty(const std::string& path);
void remove_if_exists(const std::string& path);

} // namespace naja::cli::sample


