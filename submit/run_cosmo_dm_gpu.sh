#!/bin/bash -l
#SBATCH -o /ptmp/uli/jax-hydro/snapshots/cosmo_dm/run256.out.%j
#SBATCH -e /ptmp/uli/jax-hydro/snapshots/cosmo_dm/run256.err.%j
#SBATCH -D /raven/u/uli/jax-hydro
#SBATCH -J cosmo_dm256
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=120000
#SBATCH --time=01:00:00
module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
export JAX_PLATFORM_NAME=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export LUNA_DM_IC=/ptmp/uli/jax-hydro/music/ics_arepo_l8.hdf5
export LUNA_DM_MESH=256
export LUNA_DM_AEND=1.0
cd /raven/u/uli/jax-hydro
echo "=== DM N-body 256^3 -> z=0 on 1x A100 ==="
python scripts/cosmo/run_cosmo_dm.py
echo "=== done ==="
