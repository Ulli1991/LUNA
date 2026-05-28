#!/bin/bash -l
#SBATCH -o ./wb_hse.out.%j
#SBATCH -e ./wb_hse.err.%j
#SBATCH -D ./
#SBATCH -J wb_hse
#SBATCH --partition=gpudev
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=60000
#SBATCH --time=00:05:00

module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
export JAX_PLATFORM_NAME=cuda

python -u -m scripts.validation.plots.wb_hse_test
