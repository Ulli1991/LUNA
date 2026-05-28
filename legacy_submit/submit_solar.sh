#!/bin/bash -l
#SBATCH -o ./solar.out.%j
#SBATCH -e ./solar.err.%j
#SBATCH -D ./
#SBATCH -J luna_solar
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

python -u -m scripts.validation.plots.solar_convection_run
