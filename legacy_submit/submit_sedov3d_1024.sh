#!/bin/bash -l
#SBATCH -o ./sedov.out.%j
#SBATCH -e ./sedov.err.%j
#SBATCH -D ./
#SBATCH -J luna_sedov1024
#SBATCH --partition=gpu1
#SBATCH --nodes=4
#SBATCH --ntasks=16                    # 1 task per GPU (4 per node)
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=125000
#SBATCH --time=01:30:00
#SBATCH --mail-type=BEGIN,FAIL
#SBATCH --mail-user=uli@mpa-garching.mpg.de

module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate

export JAX_PLATFORM_NAME=cuda
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Workaround XLA regressions at 1024^3 multi-node:
#   - combiner annotator overflow (negative threshold from int math)
#   - autotuner "No valid config" failure on oversized sharded transposes
# Disable autotuner + async collectives + keep combiner thresholds modest.
export XLA_FLAGS="--xla_gpu_autotune_level=0 \
                  --xla_gpu_all_reduce_combine_threshold_bytes=33554432 \
                  --xla_gpu_all_gather_combine_threshold_bytes=33554432 \
                  --xla_gpu_reduce_scatter_combine_threshold_bytes=33554432"

# JAX coordinator: use IP address of the head node (not hostname) so the
# coordination service is reachable from every worker. Pattern copied from
# the validated scripts/submit_jax_multi.sh.
NODES=($(scontrol show hostname $SLURM_JOB_NODELIST))
HEAD_NODE=${NODES[0]}
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)
# Ephemeral-range port derived from job id so concurrent multi-node runs
# don't clash on the same coordinator port.
COORD_PORT=$((49152 + SLURM_JOB_ID % 15000))
export JAX_COORDINATOR_ADDRESS="${HEAD_NODE_IP}:${COORD_PORT}"
echo "JAX_COORDINATOR_ADDRESS: $JAX_COORDINATOR_ADDRESS"

export LUNA_SEDOV_NX=1024

srun python -u -m scripts.validation.plots.sedov3d_run
