#!/bin/bash -l
#SBATCH -o ./sqdbg1.out.%j
#SBATCH -e ./sqdbg1.err.%j
#SBATCH -D ./
#SBATCH -J luna_sqdbg1
#SBATCH --partition=gpudev
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#SBATCH --time=00:10:00
module purge && module load gcc/13 cuda/12 && source ~/jax-env/bin/activate
export CUDA_VISIBLE_DEVICES=0; export JAX_PLATFORM_NAME=cuda
python -u -m scripts.validation.plots.square_dbg
