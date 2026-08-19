#!/bin/bash -l
# One weak-scaling run: fixed 512^3 per-GPU tile, Sedov 3D.
# Submit with: sbatch --nodes=<N> --export=ALL,NX=<nx>,MESH=<a,b,c> weak_run.sh
# Rungs: NX=1024 MESH=2,2,2 nodes=2 | NX=2048 MESH=4,4,4 nodes=16 |
#        NX=4096 MESH=8,8,8 nodes=128
# Per run: fp32 timed (control), recon32 timed, fp64-plain 3-step gate
# (uint32-wrap bug check on 512^3 shards — see RESULTS.md).
#SBATCH -o /e/scratch/luna/uli/runs/weak/slurm.%j.out
#SBATCH -J luna_weak_run
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
    # venv+repo from node-local /dev/shm (GPFS import-storm fix; see
    # prep_shm_env.sh). Frontier runs: startup was ~14 min at 4096 ranks.
    source "$REPO/scaling/jupiter/prep_shm_env.sh"
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
OUT=/e/scratch/luna/uli/runs/weak
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
    # shm mode keeps the node-local cache from prep_shm_env.sh
    [ "${LUNA_SHM_ENV:-0}" = 1 ] \
        || export LUNA_XLA_CACHE_DIR=/e/scratch/luna/uli/xla_cache/sedov_${tag}
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
            && echo "NaN   $tag (uint32-wrap gate: BUG PRESENT on shards?)" \
            || echo "FAILED $tag (exit $rc)"
    else
        grep "perf =" "$OUT/${tag}.log" | tail -2 | head -1
    fi
}

# STEPS: timed steps per variant (plateau needs ~3 perf windows = 30+).
# GATE=0 skips the fp64 3-step gate (per-shard program is identical at every
# run — the gate has passed at 8/64/512 GPUs; NEVER RAN at 4096 (job 1264426 cancelled first); skip on frontier runs
# where every extra srun costs thousands of node-minutes of bootstrap).
STEPS=${STEPS:-100}
GATE=${GATE:-1}
# MODES: which variants this job runs. Frontier runs run ONE mode per job
# with a hard 5-min wall — bounds the billed cost; a variant that doesn't
# fit teaches us the true startup cost for the price of 5 node-minutes each.
MODES=${MODES:-"fp32 recon32"}
for m in $MODES; do run_variant $m $STEPS; done
[ "$GATE" = "1" ] && run_variant fp64 3
echo "=== run done $(date +%T) ==="
