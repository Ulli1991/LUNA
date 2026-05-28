#!/bin/bash -l
# Multi-node JAX smoke test: VALIDATED PATTERN + XLA_FLAGS to disable the
# suspect collective combiner pass that returned a negative threshold.
#SBATCH -o ./multi_xla.out.%j
#SBATCH -e ./multi_xla.err.%j
#SBATCH -D ./
#SBATCH -J jax_xla
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

# Disable the collective combiner pass that overflowed in the previous attempt.
# Also clamp the per-combiner byte threshold to a safe positive constant in case
# the auto-tuner is the culprit.
export XLA_FLAGS="--xla_gpu_enable_all_gather_combine_by_dim=false \
                  --xla_gpu_enable_reduce_scatter_combine_by_dim=false \
                  --xla_gpu_all_reduce_combine_threshold_bytes=33554432 \
                  --xla_gpu_all_gather_combine_threshold_bytes=33554432 \
                  --xla_gpu_reduce_scatter_combine_threshold_bytes=33554432"

NODES=($(scontrol show hostname $SLURM_JOB_NODELIST))
HEAD_NODE=${NODES[0]}
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)
JAX_COORDINATOR_PORT=12345
export JAX_COORDINATOR_ADDRESS="${HEAD_NODE_IP}:${JAX_COORDINATOR_PORT}"

echo "Head: $HEAD_NODE ($HEAD_NODE_IP)"
echo "Coordinator: $JAX_COORDINATOR_ADDRESS"
echo "XLA_FLAGS: $XLA_FLAGS"

export LUNA_SEDOV_NX=128
srun python -u -m scripts.validation.plots.sedov3d_run
