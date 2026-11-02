#!/bin/bash -l
# One gravity weak-scaling run: monofonIC IC, blocks-decomposed PM N-body,
# ~40 timed steps (a 0.02 -> 0.0445, dloga_max 0.02), no snapshot.
# Submit: LEVEL=10 sbatch --nodes=2 gravity_run.sh   (NX=2^LEVEL, 4 GPUs/node)
# Site lines: set the #SBATCH -o path and --account for your site; export
# REPO (your LUNA checkout) and LUNA_SCRATCH (your scratch directory).
#SBATCH -o /path/to/scratch/runs/grav_weak/slurm.%j.out
#SBATCH -J luna_grav_run
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
NX=$((1 << LEVEL))
NG=${NG:-$((SLURM_JOB_NUM_NODES * 4))}     # override for the 1-GPU anchor
GRES_OPT=$([ "$NG" -eq 1 ] && echo "--gres=gpu:1")   # slurm overrides CUDA_VISIBLE_DEVICES; pin via the step gres
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)" hostname --ip-address)
export JAX_COORDINATOR_ADDRESS="${HEAD_IP}:$((20000 + (SLURM_JOB_ID * 7) % 40000))"
export LUNA_JAX_INIT_TIMEOUT=280
IC=$LUNA_SCRATCH/ics/ic_l${LEVEL}_${NX}.hdf5
AEND=${AEND:-0.0445}
CAP=${CAP:-1.2}                # lean start; occupancy-triggered regrow adds headroom on demand
IP=""                          # the layout is blocks on >1 GPU, dense on 1 (setup/mesh.py)
export LUNA_XLA_CACHE_DIR=$LUNA_SCRATCH/xla_cache/grav_l${LEVEL}_g${NG}${IP}
cd "${REPO:?set REPO to your LUNA checkout}"
OUT=$LUNA_SCRATCH/runs/grav_weak
mkdir -p "$OUT"
echo "=== grav run l$LEVEL (${NX}^3) ${NG} GPUs $(date +%T) ==="
srun $GRES_OPT -n "$NG" python -m scripts.cosmo.run_cosmo_dm ic=$IC mesh=$NX \
    a_end=$AEND dm_capacity_factor=$CAP write_snapshots=0 \
    > "$OUT/l${LEVEL}_g${NG}${IP}.log" 2>&1
grep -E "cosmo-dm.*(particles|done|decomposition)" "$OUT/l${LEVEL}_g${NG}${IP}.log" | tail -4
