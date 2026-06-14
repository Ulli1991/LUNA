#!/bin/bash -l
#SBATCH -o slurm_logs/m82ad4g.out.%j
#SBATCH -e slurm_logs/m82ad4g.err.%j
#SBATCH -D ./
#SBATCH -J luna_m82ad4g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=480000
#SBATCH --gres=gpu:a100:4
#SBATCH --time=15:00
module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
export JAX_PLATFORM_NAME=cuda
# single process, all 4 GPUs visible -> auto-sharding (no coordinator)
unset JAX_COORDINATOR_ADDRESS
python -u -m scripts.validation.runs.m82_autodiff_run
