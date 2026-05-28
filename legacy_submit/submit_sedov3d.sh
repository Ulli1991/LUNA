#!/bin/bash -l
#SBATCH -o ./sedov.out.%j
#SBATCH -e ./sedov.err.%j
#SBATCH -D ./
#SBATCH -J luna_sedov
#SBATCH --partition=gpudev
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=60000
#SBATCH --time=00:15:00
#SBATCH --mail-type=BEGIN,FAIL
#SBATCH --mail-user=uli@mpa-garching.mpg.de

module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
export JAX_PLATFORM_NAME=cuda

python -u -m scripts.validation.plots.sedov3d_run
