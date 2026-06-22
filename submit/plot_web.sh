#!/bin/bash -l
#SBATCH -o /ptmp/uli/jax-hydro/snapshots/cosmo_hydro/plotweb.out.%j
#SBATCH -e /ptmp/uli/jax-hydro/snapshots/cosmo_hydro/plotweb.err.%j
#SBATCH -D /raven/u/uli/jax-hydro
#SBATCH -J plot_web
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=00:10:00
module purge; module load gcc/13; source ~/jax-env/bin/activate
export JAX_PLATFORM_NAME=cpu; export PYTHONUNBUFFERED=1
cd /raven/u/uli/jax-hydro
python -u scripts/cosmo/plot_cosmic_web.py "${WEB_SNAP}" "${WEB_OUT}"
echo "rc=$?"
