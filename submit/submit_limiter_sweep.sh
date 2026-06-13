#!/bin/bash -l
#SBATCH -o slurm_logs/limiter_sweep.out.%j
#SBATCH -e slurm_logs/limiter_sweep.err.%j
#SBATCH -D ./
#SBATCH -J luna_lim_sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=60000
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
cd /raven/u/uli/jax-hydro
export XLA_FLAGS="--xla_gpu_autotune_level=1"
python -u -m scripts.autodiff.limiter_sweep
