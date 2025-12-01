#!/bin/bash -l
# JAX Native Sharding - Multi-node A100 on Raven
#SBATCH -o ./job_multi.out.%j
#SBATCH -e ./job_multi.err.%j
#SBATCH -D ./
#SBATCH -J jax_multi
#
# ===================================================================
# Multi-node configuration for JAX sharding (NO MPI!)
# ===================================================================
#SBATCH --nodes=2               # Number of nodes
#SBATCH --ntasks-per-node=1     # ONE task per node (each sees all local GPUs)
#SBATCH --cpus-per-task=72      # Full node resources
#SBATCH --mem=500000            # Full node memory
#SBATCH --constraint="gpu"      # A100 GPU nodes
#SBATCH --gres=gpu:4            # All 4 GPUs per node
#
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,FAIL
#SBATCH --mail-user=uli@mpa-garching.mpg.de

# Load modules (NO MPI needed!)
module purge
module load gcc/13 cuda/12

# Activate JAX environment
source ~/venv_jax_cuda/bin/activate  # Adjust to your venv name

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# CRITICAL: Ensure all 4 GPUs are visible to JAX on each node
export CUDA_VISIBLE_DEVICES=0,1,2,3

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
echo "Total nodes: ${#NODES[@]}"
echo "GPUs per node: 4"
echo "Total GPUs: $((${#NODES[@]} * 4))"
echo "==================================================================="

# Set JAX coordinator address
JAX_COORDINATOR_PORT=12345
export JAX_COORDINATOR_ADDRESS="${HEAD_NODE_IP}:${JAX_COORDINATOR_PORT}"

# Launch - each task sees all 4 local GPUs
srun python main_multi.py

echo "Simulation completed!"
