#!/bin/bash -l
#SBATCH -o slurm_logs/m82dt.out.%j
#SBATCH -e slurm_logs/m82dt.err.%j
#SBATCH -D ./
#SBATCH -J luna_m82dt
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
unset JAX_COORDINATOR_ADDRESS
export LUNA_M82_NX=32
export LUNA_M82_SELFGRAV=0
export LUNA_M82_GRADMODE=forward
export LUNA_M82_SHARDMAP=0
python -u -m scripts.validation.runs.m82_autodiff_run
