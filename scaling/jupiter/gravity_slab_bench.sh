#!/bin/bash -l
# Multi-GPU slab-kernel roofline (see gravity_slab_roofline.py).
# Submit: NX=1024 sbatch --nodes=2 gravity_slab_bench.sh
#SBATCH -o /e/scratch/luna/uli/runs/grav_weak/slab_bench.%j.out
#SBATCH -J luna_slab_bench
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --account=luna
#SBATCH --time=00:05:00

set -u
module load Python/3.13.5 CUDA/13
source /e/scratch/luna/uli/jax_env_cuda/bin/activate
export PYTHONUNBUFFERED=1 JAX_PLATFORM_NAME=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
NG=$((SLURM_JOB_NUM_NODES * 4))
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)" hostname --ip-address)
export JAX_COORDINATOR_ADDRESS="${HEAD_IP}:$((20000 + (SLURM_JOB_ID * 7) % 40000))"
export LUNA_JAX_INIT_TIMEOUT=280
export LUNA_GRAV_NX=${NX:?}
export LUNA_GRAV_INTPOS=${LUNA_GRAV_INTPOS:-0}   # uint32 integer positions (R4)
export LUNA_XLA_CACHE_DIR=/e/scratch/luna/uli/xla_cache/slab_bench_nx${NX}_g${NG}_ip${LUNA_GRAV_INTPOS}
cd /e/home/jusers/steinwandel2/jupiter/scaling_luna/LUNA
echo "=== slab bench nx=$NX ${NG} GPUs $(date +%T) ==="
LOG=/e/scratch/luna/uli/runs/grav_weak/slab_bench_nx${NX}_g${NG}$([ "$LUNA_GRAV_INTPOS" = 1 ] && echo _intpos).log
srun -n "$NG" python -m scripts.validation.runs.gravity_slab_roofline > "$LOG" 2>&1
grep slab-roofline "$LOG"
