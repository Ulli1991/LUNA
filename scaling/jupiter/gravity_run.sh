#!/bin/bash -l
# One gravity weak-scaling run: monofonIC IC, blocks-decomposed PM N-body,
# ~40 timed steps (a 0.02 -> 0.0445, dloga_max 0.02), no snapshot.
# Submit: LEVEL=10 sbatch --nodes=2 gravity_run.sh   (NX=2^LEVEL, 4 GPUs/node)
#SBATCH -o /e/scratch/luna/uli/runs/grav_weak/slurm.%j.out
#SBATCH -J luna_grav_run
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
NX=$((1 << LEVEL))
NG=${NG:-$((SLURM_JOB_NUM_NODES * 4))}     # override for the 1-GPU anchor
GRES_OPT=$([ "$NG" -eq 1 ] && echo "--gres=gpu:1")   # slurm overrides CUDA_VISIBLE_DEVICES; pin via the step gres
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)" hostname --ip-address)
export JAX_COORDINATOR_ADDRESS="${HEAD_IP}:$((20000 + (SLURM_JOB_ID * 7) % 40000))"
export LUNA_JAX_INIT_TIMEOUT=280
export LUNA_DM_IC=/e/scratch/luna/uli/ics/ic_l${LEVEL}_${NX}.hdf5
export LUNA_DM_MESH=$NX
export LUNA_DM_AEND=${LUNA_DM_AEND:-0.0445}
export LUNA_DM_WRITE_SNAP=0
export LUNA_DM_HOST_EVERY=${LUNA_DM_HOST_EVERY:-10}   # amortize ~10-25 ms/step GH200 host syncs
export LUNA_DM_FIXED=${LUNA_DM_FIXED:-1}              # fixed-point int32 deposit (float atomics are the slow kind)
export LUNA_DM_INTPOS=${LUNA_DM_INTPOS:-0}            # uint32 integer positions (R4, GADGET-4/pmwd-style); flip after gate
export LUNA_DM_CAP=${LUNA_DM_CAP:-1.2}                # lean start; occupancy-triggered regrow adds headroom on demand
[ "$NG" -gt 1 ] && export LUNA_DM_SLAB=${LUNA_DM_SLAB:-1}   # sharded slab path + chunked per-rank IC read
IP=$([ "${LUNA_DM_BLOCKS:-0}" = 1 ] && echo -n _blocks; [ "$LUNA_DM_INTPOS" = 1 ] && echo _intpos)
export LUNA_XLA_CACHE_DIR=/e/scratch/luna/uli/xla_cache/grav_l${LEVEL}_g${NG}${IP}
cd /e/home/jusers/steinwandel2/jupiter/scaling_luna/LUNA
OUT=/e/scratch/luna/uli/runs/grav_weak
mkdir -p "$OUT"
echo "=== grav run l$LEVEL (${NX}^3) ${NG} GPUs $(date +%T) ==="
srun $GRES_OPT -n "$NG" python -m scripts.cosmo.run_cosmo_dm > "$OUT/l${LEVEL}_g${NG}${IP}.log" 2>&1
grep -E "cosmo-dm.*(particles|done|decomposition)" "$OUT/l${LEVEL}_g${NG}${IP}.log" | tail -4
