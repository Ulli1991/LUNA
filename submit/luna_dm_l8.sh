#!/bin/bash -l
#SBATCH -o /ptmp/uli/jax-hydro/music/luna_dm_l8.out.%j
#SBATCH -e /ptmp/uli/jax-hydro/music/luna_dm_l8.err.%j
#SBATCH -D /raven/u/uli/jax-hydro
#SBATCH -J luna_dm_l8
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=120000
#SBATCH --time=00:30:00
module purge
module load gcc/13
source ~/jax-env/bin/activate
export JAX_PLATFORM_NAME=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export LUNA_DM_IC=/ptmp/uli/jax-hydro/music/ics_arepo_l8.hdf5
export LUNA_DM_MESH=256
cd /raven/u/uli/jax-hydro
echo "=== LUNA PM gravity on the 256^3 MUSIC DM IC ==="
python scripts/cosmo/run_dm_music.py
echo "=== done ==="
