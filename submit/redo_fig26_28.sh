#!/bin/bash -l
#SBATCH -o /ptmp/uli/jax-hydro/snapshots/redo_fig2628.out.%j
#SBATCH -e /ptmp/uli/jax-hydro/snapshots/redo_fig2628.err.%j
#SBATCH -D /raven/u/uli/jax-hydro
#SBATCH -J redo_fig2628
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180000
#SBATCH --time=00:20:00
module purge; module load gcc/13; source ~/jax-env/bin/activate
export JAX_PLATFORM_NAME=cpu MPLBACKEND=Agg PYTHONUNBUFFERED=1
cd /raven/u/uli/jax-hydro
echo "===== turbulence_panel (now with 1024^3) ====="
python -u -m scripts.validation.plots.turbulence_panel && echo "[turb] OK" || echo "[turb] FAILED rc=$?"
echo "===== cc85_wind_grid (OOM-fixed radial) ====="
python -u -m scripts.validation.plots.cc85_wind_grid && echo "[cc85] OK" || echo "[cc85] FAILED rc=$?"
echo "=== done ==="
