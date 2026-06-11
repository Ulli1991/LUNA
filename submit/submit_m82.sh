#!/bin/bash -l
#SBATCH -o slurm_logs/m82.out.%j
#SBATCH -e slurm_logs/m82.err.%j
#SBATCH -D ./
#SBATCH -J luna_m82
#SBATCH --partition=gpudev
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=125000
#SBATCH --time=00:15:00

module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
export JAX_PLATFORM_NAME=cuda
export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_deterministic_ops=true"
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -u -m scripts.validation.runs.m82_wind_run
