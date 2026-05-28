#!/bin/bash -l
#SBATCH -o ./sedov.out.%j
#SBATCH -e ./sedov.err.%j
#SBATCH -D ./
#SBATCH -J luna_sedov512
#SBATCH --partition=gpudev
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=125000
#SBATCH --time=00:15:00
#SBATCH --mail-type=BEGIN,FAIL
#SBATCH --mail-user=uli@mpa-garching.mpg.de

module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
export JAX_PLATFORM_NAME=cuda
export CUDA_VISIBLE_DEVICES=0,1,2,3
export LUNA_SEDOV_NX=512

python -u -m scripts.validation.plots.sedov3d_run
