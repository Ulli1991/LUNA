#!/bin/bash -l
# Phase 1 unit tests for the multigrid Poisson solver — tiny grid, single GPU.
#SBATCH -o slurm_logs/poisson_tests.out.%j
#SBATCH -e slurm_logs/poisson_tests.err.%j
#SBATCH -D ./
#SBATCH -J luna_poisson_tests
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=60000
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00

module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
export JAX_PLATFORM_NAME=cuda

python -u -m tests.test_poisson_multigrid
