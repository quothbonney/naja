#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/build"
EXE="${BUILD_DIR}/naja"

if [[ ! -x "${EXE}" ]]; then
    echo "error: ${EXE} not found. run ./scripts/build.sh first." >&2
    exit 1
fi

MODELS_DIR="${PROJECT_DIR}/models/bulk_mock"
mkdir -p "${MODELS_DIR}"

echo "preparing mock bulk dataset under ${MODELS_DIR}"

# Create a tiny mock polytope and rounding if not already present
create_mock_model() {
    local name="$1"
    local model_dir="${MODELS_DIR}/${name}"
    local gem_dir="${model_dir}/gem"
    local round_dir="${model_dir}/rounding"
    mkdir -p "${gem_dir}" "${round_dir}"

    cat > "${gem_dir}/A_eq.csv" <<'EOF'
1,0
0,1
EOF
    cat > "${gem_dir}/b_eq.csv" <<'EOF'
0
0
EOF
    cat > "${gem_dir}/l_bounds.csv" <<'EOF'
-1
-1
EOF
    cat > "${gem_dir}/u_bounds.csv" <<'EOF'
1
1
EOF

    # Simple identity rounding
    cat > "${round_dir}/${name}_rounding_A.csv" <<'EOF'
1,0
0,1
EOF
    cat > "${round_dir}/${name}_rounding_b.csv" <<'EOF'
0
0
EOF
    cat > "${round_dir}/${name}_rounding_T.csv" <<'EOF'
1,0
0,1
EOF
    cat > "${round_dir}/${name}_rounding_shift.csv" <<'EOF'
0
0
EOF
    cat > "${round_dir}/${name}_rounding_start.csv" <<'EOF'
0
0
EOF
}

JOB_LIST="${MODELS_DIR}/jobs.txt"
: > "${JOB_LIST}"
for i in $(seq 0 9); do
    name="mock_job_${i}"
    create_mock_model "${name}"
    echo "${name}" >> "${JOB_LIST}"
done

CONFIG="${MODELS_DIR}/bulk_config.txt"
cat > "${CONFIG}" <<EOF
MODEL_NAME=mock_job
DATA_DIR=${MODELS_DIR}
OUT_DIR=${MODELS_DIR}/bulk_out
N_CHAINS=2
N_SAMPLES=100
TPB_SS=64
BACK_TRANSFORM=true
WRITE_DATA=false
GPU_DEVICE=0
GPU_LIST=0
VERBOSE=false
BULK_MODEL_LIST=${JOB_LIST}
EOF

echo "config      :: ${CONFIG}"
echo "job list    :: ${JOB_LIST}"
echo

exec "${SCRIPT_DIR}/run.sh" "${CONFIG}"

