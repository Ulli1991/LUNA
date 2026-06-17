#!/bin/bash -l
#SBATCH -o /ptmp/uli/jax-hydro/snapshots/cosmo_dm/run.out.%j
#SBATCH -e /ptmp/uli/jax-hydro/snapshots/cosmo_dm/run.err.%j
#SBATCH -D /raven/u/uli/jax-hydro
#SBATCH -J cosmo_dm
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=120000
#SBATCH --time=02:00:00
module purge
module load gcc/13
source ~/jax-env/bin/activate
export JAX_PLATFORM_NAME=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export LUNA_DM_IC=${LUNA_DM_IC:-/ptmp/uli/jax-hydro/music/ics_arepo.hdf5}
export LUNA_DM_MESH=${LUNA_DM_MESH:-64}
export LUNA_DM_AEND=1.0
cd /raven/u/uli/jax-hydro
echo "=== DM N-body to z=0  IC=$LUNA_DM_IC  mesh=${LUNA_DM_MESH}^3 ==="
python scripts/cosmo/run_cosmo_dm.py
echo "=== done ==="
