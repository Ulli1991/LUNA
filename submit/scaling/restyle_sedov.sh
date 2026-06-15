#!/bin/bash -l
#SBATCH -o slurm_logs/restyle_sedov.out.%j
#SBATCH -e slurm_logs/restyle_sedov.err.%j
#SBATCH -D ./
#SBATCH -J rs_sedov
#SBATCH --partition=gpudev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --time=00:15:00
module purge; module load gcc/13
source ~/jax-env/bin/activate
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
cd /raven/u/uli/jax-hydro
python -u -m scripts.validation.plots.sedov3d_plot
