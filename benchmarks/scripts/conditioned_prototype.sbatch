#!/usr/bin/env bash
#SBATCH --partition=hpc-mid
#SBATCH --job-name=bench-conditioned
#SBATCH --time=00:30:00
#SBATCH --output=benchmarks/logs/bench-conditioned-%j.log
set -euo pipefail
B="$SLURM_SUBMIT_DIR/benchmarks"
export TMPDIR="$B/tmp" XDG_CACHE_HOME="$B/cache" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
MODE="$1"
if [[ "$MODE" == prepare ]]; then
 srun --cpu-bind=cores "$B/env-cpu/bin/python" "$B/scripts/conditioned_prototype.py" prepare
else
 shift
 STUDY_STATUS=0
 for PIPELINE in "$@"; do
  if ! srun --cpu-bind=cores "$B/env-cpu/bin/python" "$B/scripts/conditioned_prototype.py" run --pipeline "$PIPELINE" --seed "${CONDITIONED_SEED:-101}"; then
   STUDY_STATUS=1
  fi
 done
 exit "$STUDY_STATUS"
fi
