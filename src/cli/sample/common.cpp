#include "cli/sample/common.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <sys/stat.h>
#include <unistd.h>

#include "utils.h"

namespace naja::cli::sample {

[[noreturn]] void die_usage(const std::string& msg, int exit_code) {
    auto& out = (exit_code == 0) ? std::cout : std::cerr;

    if (!msg.empty()) {
        out << "error: " << msg << "\n\n";
    }

    out << "naja sample -- GPU coordinate hit-and-run sampling for polytope-constrained distributions\n";
    out << "\n";
    out << "QUICK START (bulk sampling, flat output):\n";
    out << "\n";
    out << "  # 1. Prepare models (inherit rounding from base, build feasible model list)\n";
    out << "  naja sample prepare \\\n";
    out << "    --models-root <models_root> --model-list all_models.txt \\\n";
    out << "    --base-model-dir <base_model_dir> --mode symlink \\\n";
    out << "    --out-model-list feasible.txt\n";
    out << "\n";
    out << "  # 2. Sample in bulk across multiple GPUs, writing flat .npy files\n";
    out << "  naja sample bulk \\\n";
    out << "    --models-root <models_root> --model-list feasible.txt \\\n";
    out << "    --out-root runs/ --name my_run --gpus 0,1,2,3 \\\n";
    out << "    --n-chains 24 --n-samples 2084 --tpb 256 --thinning 200 \\\n";
    out << "    --start-policy cube_center \\\n";
    out << "    --extra-constraints auto --constraint-eps 0.01 \\\n";
    out << "    --barrier-whiten --write-npy \\\n";
    out << "    --flat-output /path/to/output/samples/ \\\n";
    out << "    --skip-existing\n";
    out << "\n";
    out << "COMMANDS:\n";
    out << "  run                 sample a single model\n";
    out << "  bulk                sample many models across multiple GPUs\n";
    out << "  prepare             inherit rounding, validate, build feasible model list\n";
    out << "  inherit-rounding    copy/symlink rounding/ from one model to another\n";
    out << "  verify              validate that samples satisfy polytope constraints\n";
    out << "  eval-rounding       check rounding quality across a set of models\n";
    out << "  calibrate-rounding  fit iterative rounding pair schedule from warmup\n";
    out << "  list                list/filter models by properties\n";
    out << "\n";
    out << "KEY FLAGS (bulk):\n";
    out << "  --flat-output <dir>          write {model}.npy to a flat directory instead of\n";
    out << "                               nested run dirs (atomic via .tmp + rename)\n";
    out << "  --skip-existing              skip models already present in --flat-output\n";
    out << "                               (safe to kill and restart — resumes automatically)\n";
    out << "  --gpus <0,1,2,3>             comma-separated GPU IDs; workers steal jobs from\n";
    out << "                               a shared queue so any GPU can do any model\n";
    out << "  --start-policy cube_center   compute an LP-derived interior point per model\n";
    out << "                               (use this when no precomputed x0 exists)\n";
    out << "  --barrier-whiten             GPU eigendecomp whitening (barrier rounding) for\n";
    out << "                               better chain mixing — strongly recommended\n";
    out << "  --pair-schedule-method barrier  use barrier-Hessian Jacobi pair moves\n";
    out << "  --extra-constraints auto     load per-model extra_constraints.npy if present\n";
    out << "  --constraint-eps <eps>       slack for extra constraint feasibility (e.g. 0.01)\n";
    out << "  --thinning <n>               keep every nth step (reduces autocorrelation)\n";
    out << "  --n-chains <n>               parallel Markov chains (multiples of tpb are fastest)\n";
    out << "  --tpb <n>                    threads-per-block: 256 for V100/A100 (default 128)\n";
    out << "\n";
    out << "ROUNDING WORKFLOW:\n";
    out << "  Models need a precomputed rounding/ subdir for barrier whitening. The fast\n";
    out << "  path for large KO variant collections: calibrate once on a base/WT model,\n";
    out << "  then symlink that rounding into every KO variant with inherit-rounding.\n";
    out << "\n";
    out << "  naja sample calibrate-rounding \\\n";
    out << "    --models-root <dir> --model-list base.txt --out rounding.csv\n";
    out << "\n";
    out << "  naja sample inherit-rounding \\\n";
    out << "    --base-model-dir <base_model_dir> --target-model-dir <ko_model_dir> \\\n";
    out << "    --mode symlink\n";
    out << "\n";
    out << "  Or in bulk via prepare (does inherit + validation + feasibility check):\n";
    out << "  naja sample prepare --models-root <dir> --model-list all.txt \\\n";
    out << "    --base-model-dir <base> --mode symlink --out-model-list feasible.txt\n";
    out << "\n";
    out << "FULL SYNTAX:\n";
    out << "  naja sample run\n";
    out << "    --model-dir <dir> --out-root <dir> --gpu <id>\n";
    out << "    --n-chains <n> --n-samples <n>\n";
    out << "    [--tpb <n>] [--thinning <n>]\n";
    out << "    [--start-policy file|cube_center]\n";
    out << "    [--extra-constraints auto|ignore|require] [--extra-constraint-eps <eps>]\n";
    out << "    [--constraint-eps <eps>]\n";
    out << "    [--barrier-whiten] [--barrier-rotate]\n";
    out << "    [--pair-prob <p>] [--pair-schedule <csv>]\n";
    out << "    [--iter-rounding-passes <n>] [--iter-rounding-warmup <n>]\n";
    out << "    [--write-npy] [--bounds-policy ignore|filter] [--bounds-eps <eps>]\n";
    out << "    [--write-samples-valid] [--verbose] [--quiet] [--dry-run] [--print-config]\n";
    out << "\n";
    out << "  naja sample bulk\n";
    out << "    --models-root <dir> --model-list <file>\n";
    out << "    --out-root <dir> --name <runname> --gpus <csv>\n";
    out << "    --n-chains <n> --n-samples <n>\n";
    out << "    [--tpb <n>] [--thinning <n>]\n";
    out << "    [--start-policy file|cube_center]\n";
    out << "    [--extra-constraints auto|ignore|require] [--constraint-eps <eps>]\n";
    out << "    [--barrier-whiten] [--barrier-rotate]\n";
    out << "    [--pair-prob <p>] [--pair-schedule-method barrier]\n";
    out << "    [--flat-output <dir>] [--skip-existing]\n";
    out << "    [--write-npy] [--bounds-policy ignore|filter] [--bounds-eps <eps>]\n";
    out << "    [--write-samples-valid] [--verbose] [--quiet] [--dry-run] [--print-config]\n";
    out << "    [--resume-from <run_dir>]\n";
    out << "\n";
    out << "  naja sample prepare\n";
    out << "    --models-root <dir> --model-list <file>\n";
    out << "    [--base-model-dir <dir> --mode symlink|copy]\n";
    out << "    [--out-model-list <file>] [--out-bulk-config <file>] [--dry-run]\n";
    out << "\n";
    out << "  naja sample inherit-rounding\n";
    out << "    --base-model-dir <dir> --target-model-dir <dir> [--mode symlink|copy]\n";
    out << "\n";
    out << "  naja sample verify   --model-dir <dir>\n";
    out << "  naja sample eval-rounding\n";
    out << "    --models-root <dir> --model-list <file>\n";
    out << "    [--extra-constraint-eps <eps>] [--tight-tol <tol>]\n";
    out << "  naja sample calibrate-rounding\n";
    out << "    --models-root <dir> --model-list <file> --out <csv>\n";
    out << "    [--warmup <n>] [--passes <n>] [--pair-prob <p>]\n";
    out << "    [--extra-constraint-eps <eps>] [--seed <s>] [--max-models <n>]\n";
    out << "  naja sample list\n";
    out << "    --models-root <dir> (--prefix <p> | --glob <g>)\n";
    out << "    [--has-gem] [--has-rounding] [--out <file>]\n";
    out << "\n";
    std::exit(exit_code);
}

std::string next_arg(int& i, int argc, char** argv, const std::string& flag) {
    if (i + 1 >= argc) {
        die_usage("missing value for " + flag);
    }
    return std::string(argv[++i]);
}

void require_nonempty_file(const std::string& path, const std::string& what) {
    ::require_nonempty_file(path, what);
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


