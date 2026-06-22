#!/bin/bash -l
#SBATCH -o /ptmp/uli/jax-hydro/snapshots/redo_sedovfig.out.%j
#SBATCH -e /ptmp/uli/jax-hydro/snapshots/redo_sedovfig.err.%j
#SBATCH -D /raven/u/uli/jax-hydro
#SBATCH -J redo_sedovfig
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180000
#SBATCH --time=00:30:00
module purge; module load gcc/13; source ~/jax-env/bin/activate
export JAX_PLATFORM_NAME=cpu MPLBACKEND=Agg PYTHONUNBUFFERED=1
cd /raven/u/uli/jax-hydro
python -u -m scripts.validation.plots.sedov3d_plot && echo "[sedov] OK" || echo "[sedov] FAILED rc=$?"
