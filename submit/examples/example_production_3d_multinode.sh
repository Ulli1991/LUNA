#!/bin/bash -l
# 1024^3 Sedov 3D blast on 8 nodes / 32 GPUs.
#SBATCH -o slurm_logs/sedov.out.%j
#SBATCH -e slurm_logs/sedov.err.%j
#SBATCH -D ./
#SBATCH -J luna_sedov1024_8node
#SBATCH --nodes=8
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
echo "Coordinator: $JAX_COORDINATOR_ADDRESS"

srun python -u main.py scripts/validation/runs/sedov3d_run.py nx=1024 MESH_SHAPE=2,4,4
