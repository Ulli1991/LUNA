#!/bin/bash -l
# Grid-sharded (slab) DM N-body across 4 A100 on one node. Parameterised by
# --export: LUNA_DM_IC, LUNA_DM_MESH, LUNA_DM_OUT, LUNA_DM_AEND.
#   256^3 validation: sbatch --export=ALL,LUNA_DM_IC=.../ics_arepo_l8.hdf5,LUNA_DM_MESH=256,LUNA_DM_OUT=.../cosmo_dm_slab256 ...
#   512^3 headline   : sbatch --export=ALL,LUNA_DM_IC=.../ics_arepo_l9.hdf5,LUNA_DM_MESH=512,LUNA_DM_OUT=.../cosmo_dm_512 ...
#SBATCH -o /ptmp/uli/jax-hydro/snapshots/cosmo_dm/slab4gpu.out.%j
#SBATCH -e /ptmp/uli/jax-hydro/snapshots/cosmo_dm/slab4gpu.err.%j
#SBATCH -D /raven/u/uli/jax-hydro
#SBATCH -J cosmo_dm_slab
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=480000
#SBATCH --time=02:00:00
module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
export JAX_PLATFORM_NAME=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export PYTHONUNBUFFERED=1
export LUNA_DM_SLAB=1
export LUNA_DM_AEND=${LUNA_DM_AEND:-1.0}
export LUNA_DM_VCYC=${LUNA_DM_VCYC:-1}
cd /raven/u/uli/jax-hydro
echo "=== grid-sharded DM N-body: mesh=${LUNA_DM_MESH}^3 IC=${LUNA_DM_IC} on 4x A100 ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -u scripts/cosmo/run_cosmo_dm.py
echo "=== done ==="
