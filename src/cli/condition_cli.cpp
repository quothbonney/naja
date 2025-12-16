#include "cli/condition_cli.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "cli/condition/eflux.h"

namespace {

[[noreturn]] void die_usage(const std::string& msg) {
    std::cerr << "error: " << msg << "\n\n";
    std::cerr << "usage:\n";
    std::cerr << "  naja condition eflux <args...>\n";
    std::exit(2);
}

}

int naja_condition_cli_main(int argc, char** argv) {
    if (argc < 1) die_usage("missing subcommand");
    std::string sub = argv[0];
    if (sub == "eflux") {
        return naja::condition::cmd_eflux(argc - 1, argv + 1);
    }
    die_usage("unknown subcommand: " + sub);
}


