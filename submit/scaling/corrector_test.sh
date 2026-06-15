#!/bin/bash -l
#SBATCH -o slurm_logs/corrector_test.out.%j
#SBATCH -e slurm_logs/corrector_test.err.%j
#SBATCH -D ./
#SBATCH -J corr_test
#SBATCH --partition=gpudev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --time=00:15:00
module purge; module load gcc/13 cuda/12
source ~/jax-env/bin/activate
export JAX_PLATFORM_NAME=cuda
cd /raven/u/uli/jax-hydro
LUNA_KHC_NX=32 LUNA_KHC_NITER=120 python -u -m scripts.validation.runs.kh_corrector_run
