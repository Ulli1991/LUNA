#!/bin/bash -l
# ONE WAY to invoke tools/run_tests.py on a schedule. This file is the
# disposable, per-site half: the runner is portable, this wrapper is not.
# Copy it, edit the site lines, the two #SBATCH -o/-e paths (into OUT)
# and --account, submit it once:
#
#     sbatch submit/examples/nightly_tests.sbatch
#
# It re-arms itself at the START of the job, not the end, so a job that is
# killed, that stalls, or that never starts does not break the chain. A login
# crontab cannot do that: it lives on one login node and dies quietly when
# that node drains.
#
# One report per standard config, then --check-coverage over all of them:
#   gpu1       the full tier on ONE GPU
#   gpuN       the multidevice tier (files that ask for the device count) on
#              all NGPU GPUs of the node (2 here; on a 4-GPU node this is the
#              4-GPU config, and a 2-GPU run needs VISIBLE set to two of them)
#   dev8_fast, dev8_full   CPU, 8 virtual devices (= conftest's CPU-8 default)
#   dev2_fast  CPU, 2 virtual devices: the smallest mesh with a halo exchange
# The coverage check fails if a test is skipped in every one of them
# (allow-list: tests/skip_allowlist.json, each entry with a reason).
#
# The alarm is the ABSENCE of a fresh report, not a failure line in it:
#
#     python tools/run_tests.py --check-report --max-age-h 30
#
# exits non-zero when the report is missing, stale or failing, so a one-line
# cron/CI/manual check catches a suite that silently stopped running.
#SBATCH -J luna_tests
#SBATCH -o /path/to/scratch/luna_tests/nightly.out.%j
#SBATCH -e /path/to/scratch/luna_tests/nightly.err.%j
#SBATCH --partition=apu
#SBATCH --account=<your_account>
#SBATCH --constraint=apu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --mem=110000
#SBATCH --time=08:00:00

# ---- the site lines ------------------------------------------------------
REPO=/path/to/LUNA                    # your checkout
OUT=/path/to/scratch/luna_tests       # reports and test scratch
ENVSETUP="module load rocm/7.2 gcc/14; source ~/jax_env2/bin/activate"
EVERY="now+1day"
PLATFORM=rocm             # cuda on NVIDIA
NGPU=2                    # GPUs on the node
VISIBLE=HIP_VISIBLE_DEVICES   # CUDA_VISIBLE_DEVICES on NVIDIA
# --------------------------------------------------------------------------

mkdir -p "$OUT"
# Re-arm FIRST: whatever happens to this run, the next one is already queued.
sbatch --begin=$EVERY "$REPO/submit/examples/nightly_tests.sbatch" || true

eval "$ENVSETUP"
cd "$REPO"
# No command buffers: on 2 MI300A the second run of a shard_map program
# (replayed from a recorded HIP graph) hung the coupling worker's bridge
# kicks -- test_coupling_worker_particles and test_coupling_sources timed
# out; with the flag the same sequence runs in 10 s.
# Serialized copies: repeated block exchanges and resorts of DM particles in
# ONE process over 2 MI300A hang in the ROCm copy path (jax 0.10.2; main
# eb80392 too, once test_cosmo_dm_growth[blocks] made its particles cross
# block faces, 2026-09-30); AMD_SERIALIZE_COPY=3 runs it (rocm-crash-modes).
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export AMD_SERIALIZE_COPY=3
export LUNA_TEST_SCRATCH="$OUT/scratch"
export LUNA_DATA="${LUNA_DATA:?set LUNA_DATA to the data directory}"   # chemistry tables + Fortran references
mkdir -p "$LUNA_TEST_SCRATCH"
DAY=$(date +%Y%m%d)
R="$OUT/reports.$DAY"
mkdir -p "$R"
status=0

# GPU: the full tier on one GPU (re-times tests/.costs.local.json), the
# multidevice files on all of them.
env JAX_PLATFORMS=$PLATFORM XLA_FLAGS="--xla_gpu_enable_command_buffer=" $VISIBLE=0 \
    python tools/run_tests.py --tier full --update-costs --label gpu1 --report "$R/gpu1.json" || status=1
env JAX_PLATFORMS=$PLATFORM XLA_FLAGS="--xla_gpu_enable_command_buffer=" \
    python tools/run_tests.py --tier multidevice --label gpu$NGPU --report "$R/gpu$NGPU.json" || status=1
# CPU: --devices sets JAX_PLATFORMS=cpu and the virtual device count per file.
python tools/run_tests.py --tier fast --devices 8 --label dev8_fast --report "$R/dev8_fast.json" || status=1
python tools/run_tests.py --tier full --devices 8 --label dev8_full --report "$R/dev8_full.json" || status=1
python tools/run_tests.py --tier fast --devices 2 --label dev2_fast --report "$R/dev2_fast.json" || status=1

# The report --check-report watches, then the coverage over every config.
cp "$R/gpu1.json" "$OUT/report.json"
python tools/run_tests.py --check-coverage "$R"/gpu1.json "$R"/gpu$NGPU.json \
    "$R"/dev8_fast.json "$R"/dev8_full.json "$R"/dev2_fast.json || status=1
echo "runner exit $status"
exit $status
