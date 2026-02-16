#include "cli/sample_cli.h"

#include <string>

#include "cli/sample/common.h"
#include "cli/sample/commands.h"

int naja_sample_cli_main(int argc, char** argv) {
    if (argc < 1) naja::cli::sample::die_usage("missing subcommand");
    std::string sub = argv[0];

    if (sub == "run") {
        naja::cli::sample::cmd_run(argc - 1, argv + 1);
        return 0;
    }
    if (sub == "bulk") {
        naja::cli::sample::cmd_bulk(argc - 1, argv + 1);
        return 0;
    }
    if (sub == "verify") {
        naja::cli::sample::cmd_verify(argc - 1, argv + 1);
        return 0;
    }
    if (sub == "eval-rounding") {
        naja::cli::sample::cmd_eval_rounding(argc - 1, argv + 1);
        return 0;
    }
    if (sub == "calibrate-rounding") {
        naja::cli::sample::cmd_calibrate_rounding(argc - 1, argv + 1);
        return 0;
    }
    if (sub == "inherit-rounding") {
        naja::cli::sample::cmd_inherit_rounding(argc - 1, argv + 1);
        return 0;
    }
    if (sub == "prepare") {
        naja::cli::sample::cmd_prepare(argc - 1, argv + 1);
        return 0;
    }
    if (sub == "list") {
        naja::cli::sample::cmd_list(argc - 1, argv + 1);
        return 0;
    }

    naja::cli::sample::die_usage("unknown subcommand: " + sub);
}


