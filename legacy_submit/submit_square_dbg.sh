#!/bin/bash -l
#SBATCH -o ./sqdbg.out.%j
#SBATCH -e ./sqdbg.err.%j
#SBATCH -D ./
#SBATCH -J luna_sqdbg
#SBATCH --partition=gpudev
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=125000
#SBATCH --time=00:10:00
#SBATCH --mail-type=BEGIN,FAIL
#SBATCH --mail-user=uli@mpa-garching.mpg.de

module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=0,1,2,3
export JAX_PLATFORM_NAME=cuda

python -u -m scripts.validation.plots.square_dbg
