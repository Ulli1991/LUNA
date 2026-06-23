#!/bin/bash -l
#SBATCH -o /ptmp/uli/jax-hydro/snapshots/sm_ab.out.%j
#SBATCH -e /ptmp/uli/jax-hydro/snapshots/sm_ab.err.%j
#SBATCH -D /raven/u/uli/jax-hydro
#SBATCH -J sm_ab_256
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=36
#SBATCH --mem=120000
#SBATCH --time=00:25:00
module purge; module load gcc/13 cuda/12; source ~/jax-env/bin/activate
export JAX_PLATFORM_NAME=cuda XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONUNBUFFERED=1
export LUNA_BARY_IC=/ptmp/uli/jax-hydro/music/ics_arepo_l8_bary.hdf5
export LUNA_DM_MESH=256 LUNA_DM_DD=1 LUNA_DM_AEND=0.05 LUNA_DEBUG=1 LUNA_SNAP_EVERY=0
cd /raven/u/uli/jax-hydro
for sm in 0 1; do
  lbl=$([ $sm = 1 ] && echo "shard_map(ppermute)" || echo "auto(jnp.roll)")
  echo "================ LUNA_SHARDMAP=$sm  [$lbl] ================"
  LUNA_SHARDMAP=$sm LUNA_DM_OUT=/ptmp/uli/jax-hydro/snapshots/sm_ab_$sm \
    python -u scripts/cosmo/run_cosmo_hydro.py 2>&1 | grep -iE 'Halo exchange|shard_map with explicit|first force|s/st' | head -6
done
echo "=== done ==="
