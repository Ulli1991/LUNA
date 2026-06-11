#!/bin/bash -l
#SBATCH -o slurm_logs/turb.out.%j
#SBATCH -e slurm_logs/turb.err.%j
#SBATCH -D ./
#SBATCH -J luna_turb512
#SBATCH --partition=gpu1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=125000
#SBATCH --time=01:00:00
#SBATCH --mail-type=BEGIN,FAIL
#SBATCH --mail-user=uli@mpa-garching.mpg.de

module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
export JAX_PLATFORM_NAME=cuda
export CUDA_VISIBLE_DEVICES=0,1,2,3
export LUNA_TURB_NX=512

python -u -m scripts.validation.runs.driven_turbulence_run
