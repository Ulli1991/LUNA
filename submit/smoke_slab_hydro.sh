#!/bin/bash -l
#SBATCH -o /ptmp/uli/jax-hydro/snapshots/cosmo_hydro/smoke_slab.out.%j
#SBATCH -e /ptmp/uli/jax-hydro/snapshots/cosmo_hydro/smoke_slab.err.%j
#SBATCH -D /raven/u/uli/jax-hydro
#SBATCH -J smoke_slab_hydro
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80000
#SBATCH --time=00:30:00
module purge
module load gcc/13
source ~/jax-env/bin/activate
export JAX_PLATFORM_NAME=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=4"   # 4 simulated devices
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export LUNA_BARY_IC=/ptmp/uli/jax-hydro/music/ics_arepo_l8_bary.hdf5
export LUNA_DM_MESH=64                 # 64^3 grid, divisible by 4 slabs
export LUNA_DM_AEND=0.33               # a few KDK steps from a_init
export LUNA_DEBUG=1
export LUNA_SNAP_EVERY=0
export LUNA_DM_OUT=/ptmp/uli/jax-hydro/snapshots/cosmo_hydro
mkdir -p $LUNA_DM_OUT
cd /raven/u/uli/jax-hydro
echo "=== SLAB smoke: coupled gas+DM 64^3 on 4 simulated CPU devices ==="
LUNA_DM_DD=1 python -u scripts/cosmo/run_cosmo_hydro.py
echo "=== rc=$? ; now single-device reference (same config, 1 device) ==="
XLA_FLAGS="" LUNA_DM_DD=0 python -u scripts/cosmo/run_cosmo_hydro.py
echo "=== done rc=$? ==="
