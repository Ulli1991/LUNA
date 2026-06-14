#!/bin/bash -l
#SBATCH -o slurm_logs/inv4g.out.%j
#SBATCH -e slurm_logs/inv4g.err.%j
#SBATCH -D ./
#SBATCH -J luna_inv4g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=480000
#SBATCH --gres=gpu:a100:4
#SBATCH --time=06:00:00
module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
export JAX_PLATFORM_NAME=cuda
unset JAX_COORDINATOR_ADDRESS
export XLA_FLAGS="--xla_gpu_autotune_level=1"
srun python -u -m ${LUNA_INV_MODULE:-scripts.validation.runs.turb_autodiff_run}
