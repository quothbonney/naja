#!/usr/bin/env bash
#SBATCH --job-name=naja-f64-verify
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
set -euo pipefail
B="$PWD"
export HOME="$B/home" TMPDIR="$B/tmp" XDG_CACHE_HOME="$B/cache" PYTHONDONTWRITEBYTECODE=1
mkdir -p "$B/results/f64-build-verification"
{
 date -u +%FT%TZ
 hostname
 nvidia-smi
 lscpu
 "$B/env-gpu/bin/nvcc" --version
 sha256sum "$B/build-h100/naja" "$B/build-h100-safe/naja" "$B/build-h100-f64/naja"
} > "$B/results/f64-build-verification/HARDWARE.txt"
srun --cpu-bind=cores /usr/bin/env bash "$B/scripts/enter.sh" python "$B/scripts/verify_f64_build.py"
