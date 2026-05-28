#!/bin/bash -l
#SBATCH -o ./render.out.%j
#SBATCH -e ./render.err.%j
#SBATCH -D ./
#SBATCH -J luna_render
#SBATCH --partition=gpudev
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=125000
#SBATCH --time=00:15:00
#SBATCH --mail-type=BEGIN,FAIL
#SBATCH --mail-user=uli@mpa-garching.mpg.de

# Re-render LUNA validation figures from snapshots already on disk.
# Runs through SLURM (not login node) because stitching the 8192^2 KH set
# alone needs ~12 GB of host memory.

module purge
module load gcc/13
source ~/jax-env/bin/activate

python -u -m scripts.validation.plots.kh_grid_plot
python -u -m scripts.validation.plots.sedov3d_plot
