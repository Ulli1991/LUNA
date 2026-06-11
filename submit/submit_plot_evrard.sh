#!/bin/bash -l
#SBATCH -o slurm_logs/plot_evrard.out.%j
#SBATCH -e slurm_logs/plot_evrard.err.%j
#SBATCH -D ./
#SBATCH -J luna_plot_evrard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=60000
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00

module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate

set -e
python -u -m scripts.validation.plots.evrard_radial
ls -la figures/evrard_radial.png
