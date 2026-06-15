#!/bin/bash -l
#SBATCH -o slurm_logs/corr128noise.out.%j
#SBATCH -e slurm_logs/corr128noise.err.%j
#SBATCH -D ./
#SBATCH -J corr128n
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --time=00:45:00
module purge; module load gcc/13 cuda/12
source ~/jax-env/bin/activate
export JAX_PLATFORM_NAME=cuda
export CUDA_VISIBLE_DEVICES=0
cd /raven/u/uli/jax-hydro
# diffhydro §8.1 regime: WHITE-NOISE-seeded KH so the coarse grid genuinely
# under-resolves the turbulent shear layer. 128->1024 (8x), dt=0.001,
# train t=1 (1000), eval t=3 (3000), 50 iters to convergence.
export LUNA_KH_NOISE=0.02
LUNA_KHC_NX=128 LUNA_KHC_F=8 LUNA_KHC_DT=0.001 \
  LUNA_KHC_NSTEPS=3000 LUNA_KHC_NTRAIN=1000 LUNA_KHC_NITER=50 \
  python -u -m scripts.validation.runs.kh_corrector_run
