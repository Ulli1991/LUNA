#!/bin/bash -l
# Multi-GPU Jeans (Phase 2 shard_map self-gravity validation).
#SBATCH -o slurm_logs/jeans.out.%j
#SBATCH -e slurm_logs/jeans.err.%j
#SBATCH -D ./
#SBATCH -J luna_jeans_2node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --mem=120000
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:4
#SBATCH --time=00:15:00

module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export JAX_PLATFORM_NAME=cuda

NODES=($(scontrol show hostname $SLURM_JOB_NODELIST))
HEAD_NODE=${NODES[0]}
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)
JAX_COORDINATOR_PORT=12345
export JAX_COORDINATOR_ADDRESS="${HEAD_NODE_IP}:${JAX_COORDINATOR_PORT}"
echo "Coordinator: $JAX_COORDINATOR_ADDRESS"

export LUNA_JEANS_NX=64
export LUNA_JEANS_TEND=1.0
export LUNA_USE_SHARDMAP=1
export LUNA_MESH_SHAPE=2,2,2

srun python -u -m scripts.validation.runs.jeans_instability_run
