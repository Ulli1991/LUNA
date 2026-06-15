#!/bin/bash -l
#SBATCH -o slurm_logs/corr128np.out.%j
#SBATCH -e slurm_logs/corr128np.err.%j
#SBATCH -D ./
#SBATCH -J corr128np
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
# Fast PREVIEW of the white-noise KH IC: does coarse now under-resolve vs the
# 1024^2 reference? train t=0.5 (500), eval t=2.5 (2500), 35 iters -> fits 15min.
export LUNA_KH_NOISE=0.10
LUNA_KHC_NX=128 LUNA_KHC_F=8 LUNA_KHC_DT=0.001 \
  LUNA_KHC_NSTEPS=2500 LUNA_KHC_NTRAIN=500 LUNA_KHC_NITER=35 \
  python -u -m scripts.validation.runs.kh_corrector_run
