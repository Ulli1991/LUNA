#!/bin/bash -l
#SBATCH -o slurm_logs/grav_diag.out.%j
#SBATCH -e slurm_logs/grav_diag.err.%j
#SBATCH -D ./
#SBATCH -J luna_grav_diag
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=60000
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
python -u -m scripts.validation.plots.grav_test_diagnose
