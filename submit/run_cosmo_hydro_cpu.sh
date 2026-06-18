#!/bin/bash -l
#SBATCH -o /ptmp/uli/jax-hydro/snapshots/cosmo_hydro/cpu.out.%j
#SBATCH -e /ptmp/uli/jax-hydro/snapshots/cosmo_hydro/cpu.err.%j
#SBATCH -D /raven/u/uli/jax-hydro
#SBATCH -J cosmo_hydro_cpu
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=180000
#SBATCH --time=01:59:00
module purge
module load gcc/13
source ~/jax-env/bin/activate
export JAX_PLATFORM_NAME=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1
export LUNA_BARY_IC=/ptmp/uli/jax-hydro/music/ics_arepo_l8_bary.hdf5
export LUNA_DM_MESH=256
export LUNA_DM_AEND=${LUNA_DM_AEND:-0.2}
export LUNA_JEANS_N=${LUNA_JEANS_N:-4}
export LUNA_DM_OUT=/ptmp/uli/jax-hydro/snapshots/cosmo_hydro
mkdir -p $LUNA_DM_OUT
cd /raven/u/uli/jax-hydro
echo "=== coupled gas+DM 256^3 adiabatic -> z(a=${LUNA_DM_AEND}) on interactive CPU (72 cores) ==="
python -u scripts/cosmo/run_cosmo_hydro.py
echo "=== done ==="
