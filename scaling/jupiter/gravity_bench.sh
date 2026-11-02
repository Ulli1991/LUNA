#!/bin/bash -l
# Multi-GPU PM-kernel roofline (scripts/validation/runs/gravity_blocks_roofline.py).
# Submit: NX=1024 sbatch --nodes=2 gravity_bench.sh
# (Was the slab-kernel bench; the slab path was deleted in 274aeb5 and the
# block-sharded kernels are the only PM path.)
# Site lines: set the #SBATCH -o path and --account for your site; export
# REPO (your LUNA checkout) and LUNA_SCRATCH (your scratch directory).
#SBATCH -o /path/to/scratch/runs/grav_weak/grav_bench.%j.out
#SBATCH -J luna_grav_bench
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --account=<your_account>
#SBATCH --time=00:05:00

set -u
LUNA_SCRATCH=${LUNA_SCRATCH:?set LUNA_SCRATCH to your scratch directory}
module load Python/3.13.5 CUDA/13
source $LUNA_SCRATCH/jax_env_cuda/bin/activate
export PYTHONUNBUFFERED=1 JAX_PLATFORM_NAME=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
NG=$((SLURM_JOB_NUM_NODES * 4))
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)" hostname --ip-address)
export JAX_COORDINATOR_ADDRESS="${HEAD_IP}:$((20000 + (SLURM_JOB_ID * 7) % 40000))"
export LUNA_JAX_INIT_TIMEOUT=280
export LUNA_XLA_CACHE_DIR=$LUNA_SCRATCH/xla_cache/grav_bench_nx${NX:?}_g${NG}
cd "${REPO:?set REPO to your LUNA checkout}"
echo "=== gravity bench nx=$NX ${NG} GPUs $(date +%T) ==="
LOG=$LUNA_SCRATCH/runs/grav_weak/grav_bench_nx${NX}_g${NG}.log
srun -n "$NG" python -m scripts.validation.runs.gravity_blocks_roofline nx=$NX > "$LOG" 2>&1
grep blocks-roofline "$LOG"
