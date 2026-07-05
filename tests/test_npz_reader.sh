#!/usr/bin/env bash
# Build .npz fixtures with numpy (stored + compressed) and run the C++ reader
# against each. Skips gracefully if no python with numpy is available.
set -euo pipefail

BIN="${1:?usage: test_npz_reader.sh <path-to-test_npz_reader>}"

# Find a python that can import numpy.
PY=""
for cand in python3 \
            /data/rbg/users/jdcarson/envs/hop/bin/python3 \
            /data/rbg/users/itamarc/miniforge3/envs/dataset_prep/bin/python3; do
    if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import numpy" >/dev/null 2>&1; then
        PY="$cand"
        break
    fi
done

if [ -z "$PY" ]; then
    echo "SKIP: no python with numpy available for npz fixture generation"
    exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

"$PY" - "$TMP" <<'PYEOF'
import sys, numpy as np
d = sys.argv[1]
A = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float64)
b = np.array([7., 8.], dtype=np.float64)
start = np.array([0.1, 0.2, 0.3], dtype=np.float64)
Af = A.astype(np.float32)
np.savez(f"{d}/stored.npz", A=A, b=b, start=start, Af=Af)
np.savez_compressed(f"{d}/deflate.npz", A=A, b=b, start=start, Af=Af)
PYEOF

echo "== stored =="
"$BIN" "$TMP/stored.npz"
echo "== deflate =="
"$BIN" "$TMP/deflate.npz"
echo "test_npz_reader.sh OK"
