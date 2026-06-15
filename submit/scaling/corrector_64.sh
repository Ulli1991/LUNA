#!/bin/bash -l
#SBATCH -o slurm_logs/corr64.out.%j
#SBATCH -e slurm_logs/corr64.err.%j
#SBATCH -D ./
#SBATCH -J corr64
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
# Clean convergent KH (no noise), genuinely COARSE grid so the solver
# under-resolves the roll-up -> the CNN has deterministic, learnable structure
# to recover. 64^2 coarse -> 512^2 fine (8x). dt=0.002 (CFL 0.325 at 64^2).
# 1500 steps -> t=3.0 (eval), train to t=1.0 (step 500).
LUNA_KHC_NX=64 LUNA_KHC_F=8 LUNA_KHC_DT=0.002 \
  LUNA_KHC_NSTEPS=1500 LUNA_KHC_NTRAIN=500 LUNA_KHC_NITER=50 \
  python -u -m scripts.validation.runs.kh_corrector_run
