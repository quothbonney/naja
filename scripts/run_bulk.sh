#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

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

OUT_ROOT="${MODELS_DIR}/bulk_out"
mkdir -p "${OUT_ROOT}"

echo "job list    :: ${JOB_LIST}"
echo "out root    :: ${OUT_ROOT}"
echo

exec "${SCRIPT_DIR}/run.sh" sample bulk \
  --models-root "${MODELS_DIR}" \
  --model-list "${JOB_LIST}" \
  --out-root "${OUT_ROOT}" \
  --name mock_bulk \
  --gpus 0 \
  --n-chains 2 \
  --n-samples 100 \
  --tpb 64 \
  --bounds-policy ignore

