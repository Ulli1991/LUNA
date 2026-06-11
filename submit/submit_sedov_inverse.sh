#!/bin/bash -l
#SBATCH -o slurm_logs/sedov_inverse.out.%j
#SBATCH -e slurm_logs/sedov_inverse.err.%j
#SBATCH -D ./
#SBATCH -J luna_sedov_inv
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
python -u -m scripts.autodiff.sedov_inverse
