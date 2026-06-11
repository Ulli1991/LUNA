#!/bin/bash -l
# SMOKE TEST: 1024^3 Sedov on 4 nodes / 16 A100s — fallback if 8 GPUs OOMs.
#SBATCH -o slurm_logs/sedov_smoke.out.%j
#SBATCH -e slurm_logs/sedov_smoke.err.%j
#SBATCH -D ./
#SBATCH -J luna_sedov1024_4node_smoke
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --mem=500000
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00

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

export LUNA_SEDOV_NX=1024
# 3D block 2x2x4 at 16 GPUs: per-shard (5, 512, 512, 256) = 1.34 GB.
# At LUNA's ~28x working-set multiplier that's ~37 GB — borderline fit
# on a 40 GB A100, but the most balanced 3D mesh at 16 devices.
export LUNA_MESH_SHAPE=2,2,4
export LUNA_USE_SHARDMAP=1
export LUNA_SEDOV_OUT=/ptmp/uli/jax-hydro/snapshots/sedov_smoke_4node_Nx1024
mkdir -p "$LUNA_SEDOV_OUT"

srun python -u -m scripts.validation.runs.sedov3d_run
