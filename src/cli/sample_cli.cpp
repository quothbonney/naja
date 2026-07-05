#include "cli/sample_cli.h"

#include <array>
#include <string>
#include <utility>

#include "cli/sample/common.h"
#include "cli/sample/commands.h"

int naja_sample_cli_main(int argc, char** argv) {
    if (argc < 1) naja::cli::sample::die_usage("missing subcommand");
    const std::string sub = argv[0];
    if (sub == "--help" || sub == "-h") naja::cli::sample::die_usage("", 0);
    using CmdFn = void (*)(int, char**);
    const std::array<std::pair<const char*, CmdFn>, 8> kCommands = {{
        {"run", &naja::cli::sample::cmd_run},
        {"bulk", &naja::cli::sample::cmd_bulk},
        {"verify", &naja::cli::sample::cmd_verify},
        {"eval-rounding", &naja::cli::sample::cmd_eval_rounding},
        {"calibrate-rounding", &naja::cli::sample::cmd_calibrate_rounding},
        {"inherit-rounding", &naja::cli::sample::cmd_inherit_rounding},
        {"prepare", &naja::cli::sample::cmd_prepare},
        {"list", &naja::cli::sample::cmd_list},
    }};

    for (const auto& [name, fn] : kCommands) {
        if (sub == name) {
            fn(argc - 1, argv + 1);
            return 0;
        }
    }

    naja::cli::sample::die_usage("unknown subcommand: " + sub);
}


