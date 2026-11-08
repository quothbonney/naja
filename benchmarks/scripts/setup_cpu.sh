#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="$ROOT/cache/uv-cpu"
export UV_PYTHON_INSTALL_DIR="$ROOT/python"
export HOME="$ROOT/home"
export TMPDIR="$ROOT/tmp"
mkdir -p "$UV_CACHE_DIR" "$HOME" "$TMPDIR"
uv venv --python 3.12.15 "$ROOT/env-cpu"
uv pip install --python "$ROOT/env-cpu/bin/python" -r "$ROOT/cpu-requirements.lock.txt"
