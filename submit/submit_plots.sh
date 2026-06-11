#!/bin/bash -l
#SBATCH -o slurm_logs/plots.out.%j
#SBATCH -e slurm_logs/plots.err.%j
#SBATCH -D ./
#SBATCH -J luna_plots
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

echo "=== turbulence_panel (column density at TARGET_T=4) ==="
python -u -m scripts.validation.plots.turbulence_panel

echo "=== cc85_wind_grid (M82 wind at snapdir_060) ==="
python -u -m scripts.validation.plots.cc85_wind_grid

echo "ALL DONE"
ls -la figures/turbulence_panel.* figures/cc85_wind_*.{pdf,png} 2>/dev/null
