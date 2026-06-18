#!/bin/bash -l
#SBATCH -o /ptmp/uli/jax-hydro/snapshots/cosmo_dm/dbgfmg.out.%j
#SBATCH -e /ptmp/uli/jax-hydro/snapshots/cosmo_dm/dbgfmg.err.%j
#SBATCH -D /raven/u/uli/jax-hydro
#SBATCH -J dbg_fmg
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=200000
#SBATCH --time=00:30:00
module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
export JAX_PLATFORM_NAME=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export PYTHONUNBUFFERED=1
# set here (not via --export, whose commas separate variables)
export LUNA_DBG_NS="256"
export LUNA_DBG_NX="0"
export LUNA_DBG_NLEV="6,7"
cd /raven/u/uli/jax-hydro
echo "=== FMG shard_map compile debug: N=${LUNA_DBG_NS} on 4x A100 ==="
python -u scripts/cosmo/debug_fmg_compile.py
echo "=== done ==="
