#!/bin/bash -l
#SBATCH -o ./sq.out.%j
#SBATCH -e ./sq.err.%j
#SBATCH -D ./
#SBATCH -J luna_sq
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=60000
#SBATCH --time=00:30:00
#SBATCH --mail-type=BEGIN,FAIL
#SBATCH --mail-user=uli@mpa-garching.mpg.de

# 2D Square-advection LUNA test, single A100, 1024^2.
# One full diagonal traversal (t_end=1) on a 1x1 periodic box.

module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORM_NAME=cuda

echo "=== Square advection LUNA run, Nx=1024 ==="
nvidia-smi --query-gpu=name,memory.total --format=csv

python -u -m scripts.validation.plots.square_advection
