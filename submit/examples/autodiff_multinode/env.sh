# Sourced by every adj_multinode job: the stack, the flags, the coordinator.
# ADJ_VENV / ADJ_CMDBUF / ADJ_HLB select a variant; the defaults are the recipe
# (docs/AUTODIFF.md, "Across nodes"). Submit from the repo root, after
# mkdir -p slurm_logs.
module purge; module load rocm/7.2 gcc/14
source ${ADJ_VENV:-$HOME/venv_jax_rocm7.2_j1011}/bin/activate
WT=${ADJ_TREE:-$SLURM_SUBMIT_DIR}
export PYTHONPATH=$WT/src:$WT PYTHONUNBUFFERED=1
export JAX_PLATFORMS=rocm XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${ADJ_MEM_FRACTION:-0.95}
# Scratch for references, figures and the compile cache (compute nodes on
# viper cannot write ptmp1). Logs go to slurm_logs/ in the submit directory.
export ADJ_OUT=${ADJ_OUT:?set ADJ_OUT to a scratch directory}
mkdir -p "$ADJ_OUT"
export LUNA_XLA_CACHE_DIR=$ADJ_OUT/xla_cache_$SLURM_JOB_ID
# XLA command buffers off unless ADJ_CMDBUF=on: a shard_map program replayed
# from a recorded command buffer hangs on its second execution.
if [ "${ADJ_CMDBUF:-off}" = off ]; then
    export XLA_FLAGS=--xla_gpu_enable_command_buffer=
else
    unset XLA_FLAGS
fi
if [ "${ADJ_HLB:-0}" = 1 ]; then export HIP_LAUNCH_BLOCKING=1; else unset HIP_LAUNCH_BLOCKING; fi
# Serialize the runtime's copies, not the kernel launches: closes the race
# that crashes long gradients (cooling > ~60 steps; a small gravity
# inversion); +8% on the gravity inversion, free for the hydro. ADJ_SCOPY=0
# turns it off.
if [ "${ADJ_SCOPY:-1}" = 1 ]; then export AMD_SERIALIZE_COPY=3; else unset AMD_SERIALIZE_COPY; fi
NODES_A=($(scontrol show hostname "$SLURM_JOB_NODELIST"))
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "${NODES_A[0]}" hostname --ip-address)
export JAX_COORDINATOR_ADDRESS="${HEAD_IP}:$(( 28000 + SLURM_JOB_ID % 20000 ))"
cd "$WT" || exit 1

# srun with a fresh coordinator port, relaunched (twice) when it dies within
# a minute: with AMD_SERIALIZE_COPY=3 a process now and then segfaults while
# the ROCm client starts (3 of ~18 launches here, 0 of ~35 without).
adj_srun() {
    local attempt rc t0
    for attempt in 1 2 3; do
        export JAX_COORDINATOR_ADDRESS="${HEAD_IP}:$(( 29000 + SLURM_JOB_ID % 20000 + RANDOM % 900 ))"
        t0=$(date +%s)
        srun "$@"; rc=$?
        { [ $rc = 0 ] || [ $(( $(date +%s) - t0 )) -gt 60 ]; } && return $rc
        echo "start-up failure (rc=$rc), relaunching" >&2
    done
    return $rc
}
echo "== $(date) nodes=$SLURM_JOB_NODELIST tree=$WT venv=${ADJ_VENV:-j1011}" \
     "cmdbuf=${ADJ_CMDBUF:-off} HLB=${ADJ_HLB:-0} SCOPY=${ADJ_SCOPY:-1} coord=$JAX_COORDINATOR_ADDRESS"
