#!/bin/bash -l
#SBATCH -o ./kh.out.%j
#SBATCH -e ./kh.err.%j
#SBATCH -D ./
#SBATCH -J luna_kh
#SBATCH --partition=gpu1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=125000
#SBATCH --gres=gpu:a100:4
#SBATCH --time=01:00:00
#SBATCH --mail-type=BEGIN,FAIL
#SBATCH --mail-user=uli@mpa-garching.mpg.de

# 2D Kelvin-Helmholtz single-resolution LUNA run, 4 GPUs (1 node).
# Pass the resolution via env var when sbatch'ing:
#   sbatch --export=ALL,LUNA_KH_NX=1024 submit_validation_2d_kh.sh
#   sbatch --export=ALL,LUNA_KH_NX=2048 submit_validation_2d_kh.sh
#   sbatch --export=ALL,LUNA_KH_NX=4096 submit_validation_2d_kh.sh
#   sbatch --export=ALL,LUNA_KH_NX=8192 submit_validation_2d_kh.sh

module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=0,1,2,3
export JAX_PLATFORM_NAME=cuda

echo "=== KH LUNA run, Nx=${LUNA_KH_NX} ==="
nvidia-smi --list-gpus

python -u -m scripts.validation.plots.kh_run
