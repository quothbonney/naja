#!/usr/bin/env bash
#SBATCH --partition=hpc-mid
#SBATCH --job-name=hopsy-native-build
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=benchmarks/logs/hopsy-native-build-%j.log
set -euo pipefail
ROOT="$SLURM_SUBMIT_DIR/benchmarks"
VARIANT="${1:-native}"
if [[ "$VARIANT" == native ]]; then
  ENV=env-cpu-native
  SOURCE=source/hopsy-1.7.0
  WHEELS=wheels-native
  PREFIX=hopsy_native
  FLAGS='-O3 -march=native -mtune=native'
else
  ENV=env-cpu-sdist
  SOURCE=source-generic/hopsy-1.7.0
  WHEELS=wheels-source-generic
  PREFIX=hopsy_source_generic
  FLAGS='-O3'
fi
export HOME="$ROOT/home" TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache"
export PIP_CACHE_DIR="$ROOT/cache/pip-native"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CXXFLAGS="$FLAGS"
export CFLAGS="$CXXFLAGS" HOPSY_CXXFLAGS="$CXXFLAGS"
export CMAKE_BUILD_PARALLEL_LEVEL=8 CMAKE_EXPORT_COMPILE_COMMANDS=ON
export PATH="$ROOT/$ENV/bin:/usr/bin:/bin"
mkdir -p "$ROOT/$WHEELS"
date -u +%FT%TZ
hostname
g++ --version
g++ $FLAGS -Q --help=target > "$ROOT/audit/${PREFIX}_compiler_target.txt"
"$ROOT/$ENV/bin/python" -m pip wheel --no-build-isolation --no-deps \
  "$ROOT/$SOURCE" --wheel-dir "$ROOT/$WHEELS"
WHEEL=("$ROOT/$WHEELS"/hopsy-1.7.0-*.whl)
"$ROOT/$ENV/bin/python" -m pip install --no-deps --force-reinstall "${WHEEL[0]}"
sha256sum "${WHEEL[0]}" > "$ROOT/audit/${PREFIX}_wheel.sha256"
sha256sum "$ROOT/$ENV"/lib/python3.12/site-packages/hopsy/core*.so \
  > "$ROOT/audit/${PREFIX}_module.sha256"
"$ROOT/$ENV/bin/python" "$ROOT/scripts/native_cube_check.py" \
  --output "$ROOT/audit/${PREFIX}_cube.json"
date -u +%FT%TZ
