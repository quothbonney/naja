#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXE="${ROOT}/build/naja"

if [[ ! -x "${EXE}" ]]; then
  echo "missing executable: ${EXE}" >&2
  exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

MODEL="${TMP}/models/MODEL_X"
mkdir -p "${MODEL}/rounding" "${MODEL}/gem" "${TMP}/out"

# Minimal required rounding artifacts (non-empty).
cat > "${MODEL}/rounding/MODEL_X_rounding_A.csv" <<'EOF'
1,0
0,1
EOF
cat > "${MODEL}/rounding/MODEL_X_rounding_b.csv" <<'EOF'
0
0
EOF
cat > "${MODEL}/rounding/MODEL_X_rounding_start.csv" <<'EOF'
0
0
EOF

OUT="$("${EXE}" sample run --model-dir "${MODEL}" --out-root "${TMP}/out" --gpu 0 --n-chains 1 --n-samples 1 --bounds-policy ignore --dry-run)"
echo "${OUT}" | grep -q "^> validate model contract$"
echo "${OUT}" | grep -q "^> write generated config$"
echo "${OUT}" | grep -q "^> dry-run: not executing sampling$"

printf 'MODEL_X\n' > "${TMP}/models/model_list.txt"
OUT="$("${EXE}" sample bulk \
  --models-root "${TMP}/models" \
  --model-list "${TMP}/models/model_list.txt" \
  --out-root "${TMP}/bulk_out" \
  --name dry_bulk \
  --gpus 0 \
  --n-chains 1 \
  --n-samples 1 \
  --pair-schedule-method barrier \
  --dry-run \
  --print-config)"
echo "${OUT}" | grep -q "^> validate model contracts$"
echo "${OUT}" | grep -q "^PAIR_SCHEDULE_METHOD=barrier$"
echo "${OUT}" | grep -q "^> dry-run: not executing bulk sampling$"

