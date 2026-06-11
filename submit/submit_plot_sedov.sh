#!/bin/bash -l
#SBATCH -o slurm_logs/plot_sedov.out.%j
#SBATCH -e slurm_logs/plot_sedov.err.%j
#SBATCH -D ./
#SBATCH -J luna_plot_sedov
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=240000
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00

module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate

set -e

echo "=== sedov3d_plot (R_s(t) + projection from highest Nx, radial overlay across all Nx) ==="
python -u -m scripts.validation.plots.sedov3d_plot

echo "ALL DONE"
ls -la figures/sedov3d.* figures/sedov3d_radial.* 2>/dev/null
