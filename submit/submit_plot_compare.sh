#!/bin/bash -l
#SBATCH -o slurm_logs/plot_compare.out.%j
#SBATCH -e slurm_logs/plot_compare.err.%j
#SBATCH -D ./
#SBATCH -J luna_plot_compare
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=240000
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00

module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate

set -e

echo "=== Sedov shardmap vs baseline radial profile comparison ==="
python -u -m scripts.validation.plots.sedov3d_compare_shardmap

ls -la figures/sedov_shardmap_vs_baseline.png
