#!/bin/bash -l
# Jeans instability — Phase 1 self-gravity acceptance test.
#SBATCH -o slurm_logs/jeans.out.%j
#SBATCH -e slurm_logs/jeans.err.%j
#SBATCH -D ./
#SBATCH -J luna_jeans
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=60000
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export JAX_PLATFORM_NAME=cuda

export LUNA_JEANS_NX=${LUNA_JEANS_NX:-64}
export LUNA_JEANS_TEND=${LUNA_JEANS_TEND:-1.2}
# Toggle shard_map self-gravity. With 1 GPU + MESH_SHAPE=(1,1,1) we exercise
# the shard_map code path (ppermute halo, pmean) without actually sharding.
export LUNA_USE_SHARDMAP=${LUNA_USE_SHARDMAP:-1}
export LUNA_MESH_SHAPE=${LUNA_MESH_SHAPE:-1,1,1}

srun python -u -m scripts.validation.runs.jeans_instability_run
