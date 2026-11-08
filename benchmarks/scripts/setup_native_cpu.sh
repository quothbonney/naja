#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HOME="$ROOT/home" TMPDIR="$ROOT/tmp" UV_CACHE_DIR="$ROOT/cache/uv-cpu"
export UV_PYTHON_INSTALL_DIR="$ROOT/python"
ENV_NAME="${1:-env-cpu-native}"
TARGET="$ROOT/$ENV_NAME"
uv venv --python "$ROOT/python/cpython-3.12.15-linux-x86_64-gnu/bin/python3.12" "$TARGET"
# Match all runtime packages, initially using the generic wheel until replaced
# by the wheel compiled on the actual compute node.
uv pip install --python "$TARGET/bin/python" -r "$ROOT/cpu-requirements.lock.txt"
uv pip install --python "$TARGET/bin/python" 'cmake==3.31.10' ninja wheel pip
LOCK="$ROOT/$ENV_NAME-requirements.lock.txt"
if [[ "$ENV_NAME" == env-cpu-native ]]; then LOCK="$ROOT/cpu-native-requirements.lock.txt"; fi
uv pip freeze --python "$TARGET/bin/python" > "$LOCK"
