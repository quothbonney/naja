#!/usr/bin/env bash
#SBATCH --job-name=naja-safe-verify
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
set -euo pipefail
B="$PWD"
export HOME="$B/home" TMPDIR="$B/tmp" XDG_CACHE_HOME="$B/cache" PYTHONDONTWRITEBYTECODE=1
srun --cpu-bind=cores /usr/bin/env bash "$B/scripts/enter.sh" python "$B/scripts/verify_safe_build.py"
