#!/bin/bash -l
#SBATCH -o /ptmp/uli/jax-hydro/snapshots/cosmo_hydro_512_z0/run.out.%j
#SBATCH -e /ptmp/uli/jax-hydro/snapshots/cosmo_hydro_512_z0/run.err.%j
#SBATCH -D /raven/u/uli/jax-hydro
#SBATCH -J cosmo_hydro_512
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=240000
#SBATCH --time=06:00:00
module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
export JAX_PLATFORM_NAME=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export PYTHONUNBUFFERED=1
# 512^3 coupled gas+DM, grid-sharded slab across 4 A100, with the density floor fix
export LUNA_BARY_IC=/ptmp/uli/jax-hydro/music/ics_arepo_l9_bary.hdf5
export LUNA_DM_MESH=512
export LUNA_DM_DD=1                    # grid-sharded slab decomposition
export LUNA_DM_AEND=${LUNA_DM_AEND:-1.0}   # target z=0; banks snapshots until the wall
export LUNA_JEANS_N=${LUNA_JEANS_N:-4}
export LUNA_RHOFLOOR=${LUNA_RHOFLOOR:-1e-3}
export LUNA_TFLOOR=${LUNA_TFLOOR:-10.0}
export LUNA_DEBUG=1
export LUNA_VMAX=2000
export LUNA_TMAX=1e9
export LUNA_FLOOR_BUF=2.0
export LUNA_SNAP_EVERY=${LUNA_SNAP_EVERY:-25}
export LUNA_DM_OUT=/ptmp/uli/jax-hydro/snapshots/cosmo_hydro_512_z0
mkdir -p $LUNA_DM_OUT
cd /raven/u/uli/jax-hydro
echo "=== coupled gas+DM 512^3 grid-sharded slab -> z(a=${LUNA_DM_AEND}) on 4x A100 ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -u scripts/cosmo/run_cosmo_hydro.py
echo "=== done ==="
