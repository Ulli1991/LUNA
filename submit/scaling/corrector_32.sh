#!/bin/bash -l
#SBATCH -o slurm_logs/corr32.out.%j
#SBATCH -e slurm_logs/corr32.err.%j
#SBATCH -D ./
#SBATCH -J corr32
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
# Clean convergent KH, deliberately coarse: 32^2 -> 256^2 (8x). The coarse grid
# badly under-resolves the roll-up so the CNN has clear, learnable structure to
# recover. dt=0.004 (CFL 0.325 at 32^2). 750 steps -> t=3 (eval), train to t=1
# (step 250). Bilinear/kh_grid figure style.
LUNA_KHC_NX=32 LUNA_KHC_F=8 LUNA_KHC_DT=0.004 \
  LUNA_KHC_NSTEPS=750 LUNA_KHC_NTRAIN=500 LUNA_KHC_NITER=150 \
  python -u -m scripts.validation.runs.kh_corrector_run
