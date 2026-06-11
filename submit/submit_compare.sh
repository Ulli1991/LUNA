#!/bin/bash -l
#SBATCH -o slurm_logs/compare.out.%j
#SBATCH -e slurm_logs/compare.err.%j
#SBATCH -D ./
#SBATCH -J luna_compare
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20000
#SBATCH --time=00:15:00

source ~/jax-env/bin/activate
cd /raven/u/uli/jax-hydro
python -u -m scripts.validation.plots.arepo_vs_luna
