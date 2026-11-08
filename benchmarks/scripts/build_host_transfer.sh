#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/bin"
exec "$ROOT/scripts/enter.sh" nvcc -O3 -std=c++17 -arch=sm_90 -Xcompiler -pthread \
  "$ROOT/tools/host_transfer.cu" -o "$ROOT/bin/host_transfer"
