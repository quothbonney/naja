#include "cli/sample/common.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <sys/stat.h>
#include <unistd.h>

#include "utils.h"

namespace naja::cli::sample {

[[noreturn]] void die_usage(const std::string& msg) {
    std::cerr << "error: " << msg << "\n\n";
    std::cerr << "usage:\n";
    std::cerr << "  naja sample run --model-dir <dir> --out-root <dir> --gpu <id> --n-chains <n> --n-samples <n> [--tpb <n>] [--thinning <n>] [--start-policy file|cube_center] [--extra-constraints auto|ignore|require] [--extra-constraint-eps <eps>] [--pair-prob <p>] [--pair-schedule <csv>] [--ksparse-prob <p>] [--ksparse-k <k>] [--iter-rounding-passes <n>] [--iter-rounding-warmup <n>] [--backmap] [--write-npy] [--bounds-policy ignore|filter] [--bounds-eps <eps>] [--write-samples-valid] [--verbose] [--quiet] [--dry-run] [--print-config]\n";
    std::cerr << "  naja sample bulk --models-root <dir> --model-list <file> --out-root <dir> --name <runname> --gpus <csv> --n-chains <n> --n-samples <n> [--tpb <n>] [--thinning <n>] [--backmap] [--write-npy] [--bounds-policy ignore|filter] [--bounds-eps <eps>] [--write-samples-valid] [--verbose] [--quiet] [--dry-run] [--print-config] [--skip-existing] [--resume-from <run_dir>]\n";
    std::cerr << "  naja sample verify --model-dir <dir> [--backmap]\n";
    std::cerr << "  naja sample eval-rounding --models-root <dir> --model-list <file> [--extra-constraint-eps <eps>] [--tight-tol <tol>]\n";
    std::cerr << "  naja sample calibrate-rounding --models-root <dir> --model-list <file> --out <csv> [--warmup <n>] [--passes <n>] [--pair-prob <p>] [--extra-constraint-eps <eps>] [--seed <s>] [--max-models <n>]\n";
    std::cerr << "  naja sample inherit-rounding --base-model-dir <dir> --target-model-dir <dir> [--mode symlink|copy]\n";
    std::cerr << "  naja sample list --models-root <dir> (--prefix <p> | --glob <g>) [--has-gem] [--has-rounding] [--out <file>]\n";
    std::cerr << "  naja sample prepare --models-root <dir> --model-list <file> [--base-model-dir <dir> --mode symlink|copy] [--out-model-list <file>] [--out-bulk-config <file>] [--dry-run]\n";
    std::exit(2);
}

std::string next_arg(int& i, int argc, char** argv, const std::string& flag) {
    if (i + 1 >= argc) {
        die_usage("missing value for " + flag);
    }
    return std::string(argv[++i]);
}

void require_nonempty_file(const std::string& path, const std::string& what) {
    if (!path_exists(path)) {
        throw std::runtime_error("missing " + what + ": " + path);
    }
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        throw std::runtime_error("cannot stat " + what + ": " + path);
    }
    if (st.st_size == 0) {
        throw std::runtime_error("empty " + what + ": " + path);
    }
}

bool file_is_empty(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        throw std::runtime_error("cannot stat: " + path);
    }
    return st.st_size == 0;
}

void remove_if_exists(const std::string& path) {
    if (path_exists(path)) {
        if (unlink(path.c_str()) != 0) {
            throw std::runtime_error("cannot remove: " + path);
        }
    }
}

} // namespace naja::cli::sample


