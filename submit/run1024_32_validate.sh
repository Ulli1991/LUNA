#!/bin/bash -l
#SBATCH -o /ptmp/uli/jax-hydro/snapshots/cosmo_hydro_1024_val/run.out.%j
#SBATCH -e /ptmp/uli/jax-hydro/snapshots/cosmo_hydro_1024_val/run.err.%j
#SBATCH -D /raven/u/uli/jax-hydro
#SBATCH -J cosmo1024_val
#SBATCH --partition=gpu
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=240000
#SBATCH --time=00:30:00
module purge; module load gcc/13 cuda/12
source ~/jax-env/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export JAX_PLATFORM_NAME=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR=cuda_malloc_async
export PYTHONUNBUFFERED=1
# multi-node JAX coordinator (head node ip)
NL=($(scontrol show hostname $SLURM_JOB_NODELIST)); HN=${NL[0]}
HNIP=$(srun --nodes=1 --ntasks=1 -w "$HN" hostname --ip-address)
export JAX_COORDINATOR_ADDRESS="${HNIP}:12345"
export LUNA_BARY_IC=/ptmp/uli/jax-hydro/music/ics_arepo_l10_bary.hdf5
export LUNA_DM_MESH=1024
export LUNA_DM_DD=1
export LUNA_DM_AEND=${LUNA_DM_AEND:-0.03}     # short validation: ~30 steps
export LUNA_JEANS_N=4
export LUNA_RHOFLOOR=1e-3
export LUNA_TFLOOR=10.0
export LUNA_DEBUG=1
export LUNA_VMAX=2000
export LUNA_TMAX=1e9
export LUNA_FLOOR_BUF=2.0
export LUNA_DM_CAP=1.15
export LUNA_DM_FP32=1
export LUNA_SNAP_EVERY=10000
export LUNA_DM_OUT=/ptmp/uli/jax-hydro/snapshots/cosmo_hydro_1024_val
mkdir -p $LUNA_DM_OUT
cd /raven/u/uli/jax-hydro
echo "=== coupled gas+DM 1024^3 slab on 32 A100 (8 nodes), multi-node VALIDATION ==="
srun python -u scripts/cosmo/run_cosmo_hydro.py
echo "=== done ==="
