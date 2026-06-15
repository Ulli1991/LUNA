#!/bin/bash -l
#SBATCH -o slurm_logs/corr128.out.%j
#SBATCH -e slurm_logs/corr128.err.%j
#SBATCH -D ./
#SBATCH -J corr128
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
# Single GPU: 1024^2 fits one A100; avoids the per-step halo all-gather that
# made the 4-GPU sharded reference build slow. n_devices=1 -> no sharding.
export CUDA_VISIBLE_DEVICES=0
cd /raven/u/uli/jax-hydro
# 128^2 coarse -> 1024^2 fine (8x gap). dt=0.001 reproduces the EXACT Courant
# number (CFL=0.325) of the stable 32^2 run (smax~5.1: KH IC has p=10, |u|<=1).
# 2000 steps reach t=2.0, train to t=0.5 (step 500). Loss converges by ~it40
# (prior run: 3.9e-6 -> 6.5e-7), so 50 iters is plenty and fits the 15-min wall.
LUNA_KHC_NX=128 LUNA_KHC_F=8 LUNA_KHC_DT=0.001 \
  LUNA_KHC_NSTEPS=2000 LUNA_KHC_NTRAIN=500 LUNA_KHC_NITER=50 \
  python -u -m scripts.validation.runs.kh_corrector_run
