#!/bin/bash -l
#SBATCH -o slurm_logs/turb.out.%j
#SBATCH -e slurm_logs/turb.err.%j
#SBATCH -D ./
#SBATCH -J luna_turb256
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
export LUNA_TURB_NX=256

python -u -m scripts.validation.runs.driven_turbulence_run
