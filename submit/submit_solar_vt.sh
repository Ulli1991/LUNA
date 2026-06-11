#!/bin/bash -l
#SBATCH -o slurm_logs/solar_vt.out.%j
#SBATCH -e slurm_logs/solar_vt.err.%j
#SBATCH -D ./
#SBATCH -J luna_solarvt
#SBATCH --partition=gpudev
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=60000
#SBATCH --time=00:15:00

module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
export JAX_PLATFORM_NAME=cuda
export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_deterministic_ops=true"

python -u -m scripts.validation.runs.solar_vt_run
