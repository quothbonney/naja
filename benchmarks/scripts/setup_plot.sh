#!/usr/bin/env bash
# Isolated plotting dependencies; never modify benchmark/baseline environments.
set -euo pipefail
B="$(cd "$(dirname "$0")/.." && pwd)"
export UV_CACHE_DIR="$B/cache/uv"
export UV_PYTHON_INSTALL_DIR="$B/python"
export UV_PYTHON_DOWNLOADS=never
export HOME="$B/home" TMPDIR="$B/tmp" XDG_CACHE_HOME="$B/cache"
PY="$B/python/cpython-3.12.15-linux-x86_64-gnu/bin/python3.12"
uv venv --python "$PY" "$B/env-plot"
if [[ -s "$B/plot-requirements.lock.txt" ]]; then
  uv pip install --python "$B/env-plot/bin/python" --only-binary :all: -r "$B/plot-requirements.lock.txt"
else
  uv pip install --python "$B/env-plot/bin/python" --only-binary :all: matplotlib
  uv pip freeze --python "$B/env-plot/bin/python" > "$B/plot-requirements.lock.txt"
fi
