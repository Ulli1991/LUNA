#!/bin/bash -l
#SBATCH -o slurm_logs/gravity_figs.out.%j
#SBATCH -e slurm_logs/gravity_figs.err.%j
#SBATCH -D ./
#SBATCH -J luna_gfigs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80000
#SBATCH --partition=small
#SBATCH --time=00:30:00

module purge
module load gcc/13
source ~/jax-env/bin/activate
cd /raven/u/uli/jax-hydro

# Evrard + Bonnor-Ebert resolution convergence (64/128/256) vs AREPO. Reads the
# existing snapshots (256^3 are multi-GB), so this runs on a compute node.
JAX_PLATFORMS=cpu python -u -m scripts.validation.plots.gravity_convergence
