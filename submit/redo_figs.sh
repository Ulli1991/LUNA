#!/bin/bash -l
#SBATCH -o /ptmp/uli/jax-hydro/snapshots/redo_figs.out.%j
#SBATCH -e /ptmp/uli/jax-hydro/snapshots/redo_figs.err.%j
#SBATCH -D /raven/u/uli/jax-hydro
#SBATCH -J redo_figs
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180000
#SBATCH --time=01:00:00
module purge; module load gcc/13; source ~/jax-env/bin/activate
export JAX_PLATFORM_NAME=cpu; export PYTHONUNBUFFERED=1
cd /raven/u/uli/jax-hydro
for s in sedov3d_plot kh_grid_plot turbulence_panel shock_panels cc85_wind_grid; do
  echo "===== $s ====="
  python -u -m scripts.validation.plots.$s && echo "[$s] OK" || echo "[$s] FAILED rc=$?"
done
echo "=== done ==="
