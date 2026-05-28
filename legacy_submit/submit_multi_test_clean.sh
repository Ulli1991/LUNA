#!/bin/bash -l
# Multi-node JAX smoke test: VALIDATED PATTERN AS-IS (port 12345, hostname IP).
#SBATCH -o ./multi_clean.out.%j
#SBATCH -e ./multi_clean.err.%j
#SBATCH -D ./
#SBATCH -J jax_clean
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --mem=500000
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:4
#SBATCH --time=00:05:00

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

echo "Head: $HEAD_NODE ($HEAD_NODE_IP)"
echo "Coordinator: $JAX_COORDINATOR_ADDRESS"

# Tiny test problem: Sedov 128 (single-node fits, but we force 2-node sharding)
export LUNA_SEDOV_NX=128
srun python -u -m scripts.validation.plots.sedov3d_run
