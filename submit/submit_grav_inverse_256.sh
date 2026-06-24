#!/bin/bash -l
#SBATCH -o slurm_logs/grav_inverse_256.out.%j
#SBATCH -e slurm_logs/grav_inverse_256.err.%j
#SBATCH -D ./
#SBATCH -J grav_inv256
#SBATCH --nodes=1
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
# 256^3 production self-gravity IC-inversion on 4 A100 in ONE process, auto-
# sharded (2x2x1, jnp.roll halos). Recovers the IC density perturbation through
# LUNA's differentiable custom-VJP FMG-Poisson self-gravity. Forward is a
# remat'd lax.scan -> bounded reverse-mode tape at 256^3.
export LUNA_AD_PROBLEM=jeans
export LUNA_AD_NX=256
export LUNA_AD_NSTEPS=${LUNA_AD_NSTEPS:-120}
export LUNA_AD_NITER=${LUNA_AD_NITER:-60}
export LUNA_AD_SHARDMAP=1
export LUNA_AD_MESH=4x1x1
srun python -u -m scripts.validation.runs.ic_inversion_run
