#!/bin/bash -l
# M82 16-node / 64 GPU — validated multi-node pattern.
#SBATCH -o slurm_logs/m82.out.%j
#SBATCH -e slurm_logs/m82.err.%j
#SBATCH -D ./
#SBATCH -J luna_m82_16node
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --mem=500000
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00

module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export JAX_PLATFORM_NAME=cuda
export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_deterministic_ops=true"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_PREALLOCATE=false

NODES=($(scontrol show hostname $SLURM_JOB_NODELIST))
HEAD_NODE=${NODES[0]}
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)
JAX_COORDINATOR_PORT=12345
export JAX_COORDINATOR_ADDRESS="${HEAD_NODE_IP}:${JAX_COORDINATOR_PORT}"
echo "Nodes: ${NODES[@]}"
echo "Coordinator: $JAX_COORDINATOR_ADDRESS"

srun python -u -m scripts.validation.runs.m82_wind_run
