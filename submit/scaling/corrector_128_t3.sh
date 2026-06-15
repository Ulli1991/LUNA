#!/bin/bash -l
#SBATCH -o slurm_logs/corr128t3.out.%j
#SBATCH -e slurm_logs/corr128t3.err.%j
#SBATCH -D ./
#SBATCH -J corr128t3
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
export CUDA_VISIBLE_DEVICES=0
cd /raven/u/uli/jax-hydro
LUNA_KHC_NX=128 LUNA_KHC_F=8 LUNA_KHC_DT=0.001 \
  LUNA_KHC_NSTEPS=3000 LUNA_KHC_NTRAIN=1000 LUNA_KHC_NITER=22 \
  python -u -m scripts.validation.runs.kh_corrector_run
