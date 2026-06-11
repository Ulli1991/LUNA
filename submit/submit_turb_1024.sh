#!/bin/bash -l
# LUNA driven turbulence 1024^3 multi-node — validated pattern (no XLA_FLAGS,
# port 12345, --mem=500000, --constraint=gpu).
#SBATCH -o slurm_logs/turb.out.%j
#SBATCH -e slurm_logs/turb.err.%j
#SBATCH -D ./
#SBATCH -J luna_turb1024
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --mem=500000
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00

module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export JAX_PLATFORM_NAME=cuda
export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_deterministic_ops=true"
# Allow XLA to use more of the A100's 40 GB and avoid pre-allocation reserve.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_PREALLOCATE=false

NODES=($(scontrol show hostname $SLURM_JOB_NODELIST))
HEAD_NODE=${NODES[0]}
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)
JAX_COORDINATOR_PORT=12345
export JAX_COORDINATOR_ADDRESS="${HEAD_NODE_IP}:${JAX_COORDINATOR_PORT}"
echo "Coordinator: $JAX_COORDINATOR_ADDRESS"

export LUNA_TURB_NX=1024
srun python -u -m scripts.validation.runs.driven_turbulence_run
