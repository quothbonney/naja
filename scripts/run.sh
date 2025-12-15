#!/usr/bin/env bash
set -euo pipefail

# Run script for naja - handles CUDA library paths

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/build"
EXE="${BUILD_DIR}/naja"

# Micromamba environment (adjust this path if needed)
ENV_PATH="${ENV_PATH:-/data/rbg/users/jdcarson/envs/hop}"
if [[ -z "${ARROW_HOME:-}" ]]; then
    if [[ -d "${PROJECT_DIR}/extern/arrow" ]]; then
        ARROW_HOME="${PROJECT_DIR}/extern/arrow"
    else
        ARROW_HOME="${PROJECT_DIR}/extern/arrow-src"
    fi
fi

if [[ ! -f "${EXE}" ]]; then
    echo "executable not found: ${EXE}" >&2
    echo "run ./scripts/build.sh first" >&2
    exit 1
fi

# Add CUDA libraries from micromamba env to LD_LIBRARY_PATH
export LD_LIBRARY_PATH="${ENV_PATH}/lib:${LD_LIBRARY_PATH:-}"

# Add PyArrow libraries
PYARROW_LIB="/data/rbg/users/jdcarson/envs/hop/lib/python3.10/site-packages/pyarrow"
if [[ -d "${PYARROW_LIB}" ]]; then
    export LD_LIBRARY_PATH="${PYARROW_LIB}:${LD_LIBRARY_PATH}"
fi

cd "${PROJECT_DIR}"

if [[ $# -gt 0 ]]; then
    CONFIG_PATH="$1"
    shift
else
    CONFIG_PATH="${PROJECT_DIR}/config.txt"
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "config not found: ${CONFIG_PATH}" >&2
    exit 1
fi

CONFIG_PATH="$(python3 - "$CONFIG_PATH" <<'PY'
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
)"

get_config_value() {
    local key="$1"
    local file="$2"
    local line
    line="$(grep -E "^[[:space:]]*${key}[[:space:]]*=" "$file" | tail -n1 | cut -d'=' -f2-)"
    echo "${line}" | tr -d '[:space:]'
}

MODEL_NAME="$(get_config_value "MODEL_NAME" "${CONFIG_PATH}")"
if [[ -z "${MODEL_NAME}" ]]; then
    MODEL_NAME="MODEL"
fi

OUT_DIR_VALUE="$(get_config_value "OUT_DIR" "${CONFIG_PATH}")"
if [[ -z "${OUT_DIR_VALUE}" ]]; then
    OUT_DIR_VALUE="./out"
fi

OUT_ABS="$(python3 - "$OUT_DIR_VALUE" "$PROJECT_DIR" <<'PY'
import os, sys
path = sys.argv[1]
base = sys.argv[2]
if not os.path.isabs(path):
    path = os.path.join(base, path)
print(os.path.abspath(path))
PY
)"

mkdir -p "${OUT_ABS}"

DATE_STR="$(date +%Y%m%d)"
RUN_INDEX=1
while true; do
    RUN_NAME="$(printf "%s_%s_%03d" "${MODEL_NAME}" "${DATE_STR}" "${RUN_INDEX}")"
    RUN_DIR="${OUT_ABS}/${RUN_NAME}"
    if [[ ! -d "${RUN_DIR}" ]]; then
        mkdir -p "${RUN_DIR}"
        break
    fi
    RUN_INDEX=$((RUN_INDEX + 1))
done

echo "config      :: ${CONFIG_PATH}"
echo "output      :: ${RUN_DIR}"
echo

export NAJA_RUN_DIR="${RUN_DIR}"
export NAJA_CONFIG_PATH="${CONFIG_PATH}"

# Run conditioning pre-processor
CONDITIONER="${PROJECT_DIR}/src/conditioning/conditioner.py"
if [[ -f "${CONDITIONER}" ]]; then
    python3 "${CONDITIONER}" "${CONFIG_PATH}"
fi

exec "${EXE}" "${CONFIG_PATH}" "$@"
