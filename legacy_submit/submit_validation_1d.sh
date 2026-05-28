#!/bin/bash -l
#SBATCH -o ./validation_1d.out.%j
#SBATCH -e ./validation_1d.err.%j
#SBATCH -D ./
#SBATCH -J luna_val_1d
#SBATCH --partition=gpudev
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#SBATCH --time=00:15:00
#SBATCH --mail-type=BEGIN,FAIL
#SBATCH --mail-user=uli@mpa-garching.mpg.de

# 1D validation suite: soundwave convergence + shock-panel comparisons.
# Single A100 — 1D problems are tiny and don't need multi-GPU.
#
# Override which entry point runs by sbatch'ing with --export, e.g.:
#   sbatch --export=ALL,VAL_TARGET=soundwave submit_validation_1d.sh
#   sbatch --export=ALL,VAL_TARGET=shock submit_validation_1d.sh   (default)

module purge
module load gcc/14 openmpi cuda/13
source ~/jax-env/bin/activate
export JAX_PLATFORM_NAME=cuda

VAL_TARGET="${VAL_TARGET:-shock}"

case "$VAL_TARGET" in
    soundwave)
        srun python -m scripts.validation.plots.soundwave_convergence
        ;;
    standing)
        srun python -m scripts.validation.plots.soundwave_standing_convergence
        ;;
    diagnostic)
        srun python -m scripts.validation.plots.soundwave_diagnostics
        ;;
    shock)
        srun python -m scripts.validation.plots.shock_panels
        ;;
    *)
        echo "Unknown VAL_TARGET=$VAL_TARGET (expected 'soundwave', 'diagnostic', or 'shock')"
        exit 1
        ;;
esac
