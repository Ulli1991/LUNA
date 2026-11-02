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
# Site lines: set the #SBATCH -o path and --account for your site; export
# REPO (your LUNA checkout) and LUNA_SCRATCH (your scratch directory).
#SBATCH -o /path/to/scratch/runs/strong/slurm.%j.out
#SBATCH -J luna_strong_run
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --account=<your_account>
#SBATCH --time=00:05:00

set -u
LUNA_SCRATCH=${LUNA_SCRATCH:?set LUNA_SCRATCH to your scratch directory}
module load Python/3.13.5 CUDA/13
REPO=${REPO:?set REPO to your LUNA checkout}
if [ "${LUNA_SHM_ENV:-0}" = 1 ]; then
    source "$REPO/scaling/jupiter/prep_shm_env.sh"   # GPFS import-storm fix
    REPO=$LUNA_REPO_DIR
else
    source $LUNA_SCRATCH/jax_env_cuda/bin/activate
fi
export PYTHONUNBUFFERED=1 JAX_PLATFORM_NAME=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
# MESH (e.g. 2,2,4) reaches the run script as MESH_SHAPE=$MESH below
cd "$REPO"

NG=$((SLURM_JOB_NUM_NODES * 4))
OUT=$LUNA_SCRATCH/runs/strong
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
    [ "${LUNA_SHM_ENV:-0}" = 1 ] \
        || export LUNA_XLA_CACHE_DIR=$LUNA_SCRATCH/xla_cache/sedov_strong_${tag}
    case $mode in
        fp32)    local prec="enable_x64=0 hydro_recon32=0" ;;
        recon32) local prec="enable_x64=1 hydro_recon32=1" ;;
        fp64)    local prec="enable_x64=1 hydro_recon32=0" ;;
    esac
    echo "=== $tag steps=$steps port=$port $(date +%T) ==="
    srun -n $NG python -m scripts.validation.runs.sedov3d_run \
        nx=$NX max_steps=$steps write_snapshots=0 MESH_SHAPE=$MESH $prec \
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
