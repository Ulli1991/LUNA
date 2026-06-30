#!/bin/bash -l
#SBATCH -o slurm_logs/grav_inverse_256.out.%j
#SBATCH -e slurm_logs/grav_inverse_256.err.%j
#SBATCH -D ./
#SBATCH -J grav_inv256
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --mem=200000
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#SBATCH --time=03:00:00
module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export JAX_PLATFORM_NAME=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR=cuda_malloc_async
export XLA_FLAGS="--xla_gpu_autotune_level=0"   # avoid autotuner "no valid config" on the big fp64 rollout transpose
# 256^3 production self-gravity IC-inversion on 8 A100 (2 nodes, 1 GPU/process),
# shard_map slab (8x1x1). 8-way sharding halves the per-GPU footprint vs the
# 4-GPU run that OOM'd (18.8 GiB). custom-VJP FMG-Poisson adjoint; remat'd
# lax.scan -> bounded reverse-mode tape.
# multi-node JAX coordinator (head node ip):
NL=($(scontrol show hostname $SLURM_JOB_NODELIST)); HN=${NL[0]}
HNIP=$(srun --nodes=1 --ntasks=1 -w "$HN" hostname --ip-address)
export JAX_COORDINATOR_ADDRESS="${HNIP}:12345"
export LUNA_AD_PROBLEM=jeans
export LUNA_AD_NX=256
export LUNA_AD_NSTEPS=${LUNA_AD_NSTEPS:-120}
export LUNA_AD_NITER=${LUNA_AD_NITER:-60}
export LUNA_AD_SHARDMAP=1
export LUNA_AD_MESH=8x1x1
srun python -u -m scripts.validation.runs.ic_inversion_run
