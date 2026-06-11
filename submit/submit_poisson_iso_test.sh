#!/bin/bash -l
#SBATCH -o slurm_logs/poisson_iso.out.%j
#SBATCH -e slurm_logs/poisson_iso.err.%j
#SBATCH -D ./
#SBATCH -J luna_pois_iso
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=125000
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#SBATCH --partition=gpudev
#SBATCH --time=00:10:00

module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
cd /raven/u/uli/jax-hydro
python -u -m tests.test_poisson_isolated
