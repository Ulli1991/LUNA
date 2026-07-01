#!/bin/bash -l
# JAX Native Sharding - Multi-node A100 on Raven
#SBATCH -o ./job_amr.out.%j
#SBATCH -e ./job_amr.err.%j
#SBATCH -D ./
#SBATCH --partition=gpu1
#SBATCH -J amr_4gpu
#
# ===================================================================
# Multi-node configuration for JAX sharding (NO MPI!)
# ===================================================================
#SBATCH --nodes=1               # single node, 4 A100s
#SBATCH --ntasks-per-node=4     # ONE task per GPU (JAX 1-GPU-per-process model)
#SBATCH --cpus-per-task=18      # 72 cpus / 4 tasks
#SBATCH --mem=500000            # Full node memory
#SBATCH --constraint="gpu"      # A100 GPU nodes
#SBATCH --gres=gpu:4            # All 4 GPUs per node
#
#SBATCH --time=01:00:00
#SBATCH --mail-type=BEGIN,FAIL
#SBATCH --mail-user=uli@mpa-garching.mpg.de

# Load modules (NO MPI needed!)
module purge
module load gcc/13 cuda/12

# Activate JAX environment
source ~/jax-env/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Let SLURM bind each task to its own GPU (1-GPU-per-process model).
# Do NOT force CUDA_VISIBLE_DEVICES=0,1,2,3 here — that would make every
# task on a node see all 4 GPUs and fight over them.

# Force CUDA platform (not ROCm!)
export JAX_PLATFORM_NAME=cuda

# ===================================================================
# JAX DISTRIBUTED CONFIGURATION
# ===================================================================
# Get coordinator info (first node)
NODES=($(scontrol show hostname $SLURM_JOB_NODELIST))
HEAD_NODE=${NODES[0]}
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)

echo "==================================================================="
echo "JAX Multi-Node Configuration"
echo "==================================================================="
echo "Head node: $HEAD_NODE ($HEAD_NODE_IP)"
echo "All nodes: ${NODES[@]}"
echo "Total nodes: ${#NODES[@]}"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total tasks: $SLURM_NTASKS"
echo "GPUs per node: 4"
echo "Total GPUs: $((${#NODES[@]} * 4))"
echo "==================================================================="

# Set JAX coordinator address
JAX_COORDINATOR_PORT=12345
export JAX_COORDINATOR_ADDRESS="${HEAD_NODE_IP}:${JAX_COORDINATOR_PORT}"

echo "JAX_COORDINATOR_ADDRESS: $JAX_COORDINATOR_ADDRESS"
echo "==================================================================="

# Verify GPU visibility on each node
echo "Verifying GPU visibility on each node:"
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c 'echo "Node $(hostname): GPUs = $(nvidia-smi --list-gpus | wc -l)"'
echo "==================================================================="

# Launch - each task sees all 4 local GPUs
# IMPORTANT: Use srun here so each node gets its own process with correct SLURM_PROCID
srun python main.py

echo "Simulation completed!"
