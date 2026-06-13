#!/bin/bash -l
#SBATCH -o slurm_logs/ou_inverse.out.%j
#SBATCH -e slurm_logs/ou_inverse.err.%j
#SBATCH -D ./
#SBATCH -J luna_ou_inv
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=60000
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00

module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
cd /raven/u/uli/jax-hydro

export LUNA_OU_NX=${LUNA_OU_NX:-128}
export LUNA_OU_NSTEPS=${LUNA_OU_NSTEPS:-250}
export LUNA_OU_ITER=${LUNA_OU_ITER:-80}

export XLA_FLAGS="--xla_gpu_autotune_level=1"
python -u -m scripts.autodiff.ou_inverse
