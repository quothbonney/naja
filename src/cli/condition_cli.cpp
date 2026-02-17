#include "cli/condition_cli.h"

#include <cstdlib>
#include <iostream>
#include <string>

#include "cli/condition/registry.h"

namespace {

[[noreturn]] void die_usage(const std::string& msg) {
    std::cerr << "error: " << msg << "\n\n";
    std::cerr << "usage:\n";
    std::cerr << "  naja condition <subcommand> <args...>\n";
    std::cerr << "\nsubcommands:\n";
    for (const auto& s : naja::condition::list_condition_subcommands()) {
        std::cerr << "  " << s << "\n";
    }
    std::exit(2);
}

}

int naja_condition_cli_main(int argc, char** argv) {
    if (argc < 1) die_usage("missing subcommand");
    std::string sub = argv[0];
    auto fn = naja::condition::lookup_condition_cmd(sub);
    if (!fn) die_usage("unknown subcommand: " + sub);
    return fn(argc - 1, argv + 1);
}




