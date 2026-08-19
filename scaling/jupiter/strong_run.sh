#!/bin/bash -l
# One strong-ladder run: fixed global NX, vary GPU count. Sedov 3D.
# Submit with: NX=<nx> MESH=<a,b,c> [MODES=...] sbatch --nodes=<N> strong_run.sh
# (NOT --export=ALL,MESH=a,b,c — sbatch splits the value on commas.)
# Curves (diagonal anchors come from the weak ladder / 1-GPU roofline):
#   NX=512:  g8  MESH=2,2,2 nodes=2  | g64  MESH=4,4,4 nodes=16
#   NX=1024: g64 MESH=4,4,4 nodes=16 | g512 MESH=8,8,8 nodes=128
#   NX=2048: g512 MESH=8,8,8 nodes=128
# Tiles here are <=256^3, below the fp64 uint32-wrap threshold (see
# RESULTS.md), so fp64-plain runs TIMED on strong runs, not just gated.
#SBATCH -o /e/scratch/luna/uli/runs/strong/slurm.%j.out
#SBATCH -J luna_strong_run
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --account=luna
#SBATCH --time=00:05:00

set -u
module load Python/3.13.5 CUDA/13
REPO=/e/home/jusers/steinwandel2/jupiter/scaling_luna/LUNA
if [ "${LUNA_SHM_ENV:-0}" = 1 ]; then
    source "$REPO/scaling/jupiter/prep_shm_env.sh"   # GPFS import-storm fix
    REPO=$LUNA_REPO_DIR
else
    source /e/scratch/luna/uli/jax_env_cuda/bin/activate
fi
export PYTHONUNBUFFERED=1 JAX_PLATFORM_NAME=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export LUNA_SEDOV_WRITE_SNAPSHOTS=0
export LUNA_MESH_SHAPE=$MESH
cd "$REPO"

NG=$((SLURM_JOB_NUM_NODES * 4))
OUT=/e/scratch/luna/uli/runs/strong
mkdir -p "$OUT"

NODES=($(scontrol show hostname $SLURM_JOB_NODELIST))
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "${NODES[0]}" hostname --ip-address)
echo "run: NX=$NX mesh=$MESH gpus=$NG head=$HEAD_IP"

IDX=0
run_variant () {  # <mode> <steps>
    local mode=$1 steps=$2
    local tag="nx${NX}_g${NG}_${mode}"
    local port=$((20000 + (SLURM_JOB_ID * 7 + IDX) % 40000)); IDX=$((IDX + 1))
    export JAX_COORDINATOR_ADDRESS="${HEAD_IP}:${port}"
    export LUNA_SEDOV_NX=$NX LUNA_SEDOV_MAX_STEPS=$steps
    [ "${LUNA_SHM_ENV:-0}" = 1 ] \
        || export LUNA_XLA_CACHE_DIR=/e/scratch/luna/uli/xla_cache/sedov_strong_${tag}
    case $mode in
        fp32)    export LUNA_X64=0 LUNA_HYDRO_RECON32=0 ;;
        recon32) export LUNA_X64=1 LUNA_HYDRO_RECON32=1 ;;
        fp64)    export LUNA_X64=1 LUNA_HYDRO_RECON32=0 ;;
    esac
    echo "=== $tag steps=$steps port=$port $(date +%T) ==="
    srun -n $NG python -m scripts.validation.runs.sedov3d_run \
        > "$OUT/${tag}.log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        grep -q "non-finite" "$OUT/${tag}.log" \
            && echo "NaN   $tag" \
            || echo "FAILED $tag (exit $rc)"
    else
        grep "perf =" "$OUT/${tag}.log" | tail -2 | head -1
    fi
}

STEPS=${STEPS:-100}
MODES=${MODES:-"fp32 recon32 fp64"}
for m in $MODES; do run_variant $m $STEPS; done
echo "=== run done $(date +%T) ==="
