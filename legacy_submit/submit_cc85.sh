#!/bin/bash -l
#SBATCH -o ./cc85.out.%j
#SBATCH -e ./cc85.err.%j
#SBATCH -D ./
#SBATCH -J luna_cc85
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
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -u -m scripts.validation.plots.cc85_wind_run
