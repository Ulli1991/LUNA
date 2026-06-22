#!/bin/bash -l
#SBATCH -o /ptmp/uli/jax-hydro/snapshots/cosmo_hydro/run.out.%j
#SBATCH -e /ptmp/uli/jax-hydro/snapshots/cosmo_hydro/run.err.%j
#SBATCH -D /raven/u/uli/jax-hydro
#SBATCH -J cosmo_hydro
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=120000
#SBATCH --time=00:30:00
module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
export JAX_PLATFORM_NAME=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export PYTHONUNBUFFERED=1
export LUNA_BARY_IC=/ptmp/uli/jax-hydro/music/ics_arepo_l8_bary.hdf5
export LUNA_DM_MESH=256
export LUNA_DEBUG=1
export LUNA_DM_AEND=${LUNA_DM_AEND:-0.30}
export LUNA_JEANS_N=${LUNA_JEANS_N:-4}
export LUNA_DM_OUT=/ptmp/uli/jax-hydro/snapshots/cosmo_hydro
mkdir -p $LUNA_DM_OUT
cd /raven/u/uli/jax-hydro
echo "=== coupled gas+DM 256^3 adiabatic -> z=${LUNA_DM_AEND} on 1x A100 ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -u scripts/cosmo/run_cosmo_hydro.py
echo "=== done ==="
