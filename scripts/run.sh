#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/build"
EXE="${BUILD_DIR}/naja"

# Runtime libraries come from the container/toolchain (CUDA) and the vendored
# Gurobi (extern/gurobi/linux64/lib, also baked into the binary's RPATH). No
# CSAIL conda env is assumed. Set ENV_PATH only if you need to prepend an extra
# lib dir (e.g. ENV_PATH=/path/to/env -> $ENV_PATH/lib on LD_LIBRARY_PATH).
ENV_PATH="${ENV_PATH:-}"
GUROBI_LIB="${PROJECT_DIR}/extern/gurobi/linux64/lib"

if [[ ! -x "${EXE}" ]]; then
    echo "executable not found: ${EXE}" >&2
    echo "run ./scripts/build.sh first" >&2
    exit 1
fi

export LD_LIBRARY_PATH="${GUROBI_LIB}:${ENV_PATH:+${ENV_PATH}/lib:}${LD_LIBRARY_PATH:-}"
cd "${PROJECT_DIR}"

if [[ $# -lt 1 ]]; then
    echo "usage: ./scripts/run.sh <sample|condition> <args...>" >&2
    exit 2
fi

if [[ "$1" != "sample" && "$1" != "condition" ]]; then
    echo "legacy config-file entrypoint is removed; use sample/condition subcommands" >&2
    exit 2
fi

exec "${EXE}" "$@"
