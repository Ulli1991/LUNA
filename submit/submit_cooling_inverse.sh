#!/bin/bash -l
#SBATCH -o slurm_logs/cooling_inverse.out.%j
#SBATCH -e slurm_logs/cooling_inverse.err.%j
#SBATCH -D ./
#SBATCH -J luna_cool_inv
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=60000
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:15:00
module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
export JAX_PLATFORM_NAME=cuda
export XLA_FLAGS="--xla_gpu_autotune_level=1"
export LUNA_AD_PROBLEM=cooling
export LUNA_AD_NX=${LUNA_AD_NX:-32}
export LUNA_AD_NSTEPS=${LUNA_AD_NSTEPS:-60}
export LUNA_AD_NITER=${LUNA_AD_NITER:-60}
export LUNA_AD_TEND_MYR=${LUNA_AD_TEND_MYR:-0.12}
srun python -u -m scripts.validation.runs.ic_inversion_run
