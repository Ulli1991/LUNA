#!/bin/bash -l
#SBATCH -o slurm_logs/sedov_ad.out.%j
#SBATCH -e slurm_logs/sedov_ad.err.%j
#SBATCH -D ./
#SBATCH -J luna_sedov_ad
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=60000
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
cd /raven/u/uli/jax-hydro
python -u -m scripts.autodiff.validate_sedov
