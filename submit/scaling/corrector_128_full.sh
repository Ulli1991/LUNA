#!/bin/bash -l
#SBATCH -o slurm_logs/corr128full.out.%j
#SBATCH -e slurm_logs/corr128full.err.%j
#SBATCH -D ./
#SBATCH -J corr128f
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --time=00:45:00
module purge; module load gcc/13 cuda/12
source ~/jax-env/bin/activate
export JAX_PLATFORM_NAME=cuda
export CUDA_VISIBLE_DEVICES=0
cd /raven/u/uli/jax-hydro
# Full diffhydro §8.1 horizon: 128^2 coarse -> 1024^2 fine (8x), dt=0.001
# (CFL 0.325). 3000 steps -> t=3.0 (eval), train to t=1.0 (step 1000) like
# Horowitz. This is the regime where the corrector's win is visible (t>=2).
LUNA_KHC_NX=128 LUNA_KHC_F=8 LUNA_KHC_DT=0.001 \
  LUNA_KHC_NSTEPS=3000 LUNA_KHC_NTRAIN=1000 LUNA_KHC_NITER=50 \
  python -u -m scripts.validation.runs.kh_corrector_run
