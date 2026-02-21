// Minimal standalone GPU sampler for RECON1 - sanity check for GPU code changes
#include <iostream>
#include <string>

#include "cli/sample_cli.h"
#include "cli/condition_cli.h"
#include "cli/validate_cli.h"

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "sample") {
        return naja_sample_cli_main(argc - 2, argv + 2);
    }
    if (argc >= 2 && std::string(argv[1]) == "condition") {
        return naja_condition_cli_main(argc - 2, argv + 2);
    }
    if (argc >= 2 && std::string(argv[1]) == "validate") {
        return naja_validate_cli_main(argc - 2, argv + 2);
    }

    std::cerr << "usage:\n";
    std::cerr << "  naja sample <subcommand> <args...>\n";
    std::cerr << "  naja condition <subcommand> <args...>\n";
    std::cerr << "  naja validate --samples <file.npy> [--model-dir <dir>] [--n-chains 4]\n";
    return 2;
}
