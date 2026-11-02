#!/bin/bash -l
#SBATCH -J adjdiag
#SBATCH -o slurm_logs/adj_diag.%j.out
#SBATCH -e slurm_logs/adj_diag.%j.err
#SBATCH --partition=apu --account=<your_account> --constraint=apu
#SBATCH --nodes=2 --ntasks-per-node=2 --cpus-per-task=24 --gres=gpu:2 --mem=200000
#SBATCH --time=00:55:00
# The multi-node bisection: each derivative program (forward, JVP, VJP) of the
# IC inversion's physics called 3 times across 2 nodes (4 GPUs, mesh 2,2,1),
# under each launch variant. A step that hangs is cut by `timeout`; its last
# "start" line names the program.
source submit/examples/autodiff_multinode/env.sh
unset AMD_SERIALIZE_COPY          # the variants set it themselves
NX=${NX:-64}; MESH=${MESH:-2,2,1}; T=${T:-420}
port=$(( 28000 + SLURM_JOB_ID % 20000 ))
run() {   # run <label> <venv> <cmdbuf on|off> <HLB 0|1> <problem> [steps] [VAR=value]
    port=$((port + 7))
    export JAX_COORDINATOR_ADDRESS="${HEAD_IP}:$port"
    source $2/bin/activate
    if [ "$3" = off ]; then export XLA_FLAGS=--xla_gpu_enable_command_buffer=; else unset XLA_FLAGS; fi
    if [ "$4" = 1 ]; then export HIP_LAUNCH_BLOCKING=1; else unset HIP_LAUNCH_BLOCKING; fi
    local steps=${6:-10} extra=${7:-ADJ_NONE=1}
    echo "#### VARIANT $1: venv=$(basename $2) cmdbuf=$3 HLB=$4 problem=$5 steps=$steps $extra $(date +%T)"
    t0=$(date +%s)
    if [ "$5" = coll ]; then
        env NCCL_DEBUG=INFO $extra timeout 240 srun python tools/adj_multinode.py coll calls=5
    else
        env $extra timeout $T srun python tools/adj_multinode.py probe problem=$5 nx=$NX n_steps=$steps mesh=$MESH calls=3
    fi
    echo "#### END $1 rc=$? after $(( $(date +%s) - t0 )) s"
}
J1011=${J1011:-$HOME/venv_jax_rocm7.2_j1011}
J1010=${J1010:-$HOME/venv_jax_rocm7.2}
for v in ${VARIANTS:-K L A B C D E F}; do
  case $v in
    K) run K $J1011 off 0 coll ;;      # bare ring ppermute + psum, async
    L) run L $J1011 off 1 coll ;;      # the same, serialized launches
    A) run A $J1011 on  0 jeans ;;     # the defaults
    B) run B $J1011 off 0 jeans ;;     # command buffers off, async launches
    C) run C $J1011 on  1 jeans ;;     # the old recipe: serialized launches
    D) run D $J1011 off 0 hydro ;;     # halo VJP alone
    E) run E $J1011 off 0 cooling ;;   # cooling + OU forcing
    F) run F $J1010 on  0 jeans ;;     # jax 0.10.0, defaults
    G) run G $J1011 off 1 jeans ;;
    H) run H $J1011 off 1 cooling ;;
    M) run M $J1011 off 0 coll 0 AMD_SERIALIZE_COPY=3 ;;     # copies serialized
    N) run N $J1011 off 0 cooling 120 AMD_SERIALIZE_COPY=3 ;;
    O) run O $J1011 off 0 cooling 120 ;;
    M2) run M2 $J1011 off 0 coll 0 "AMD_SERIALIZE_COPY=3 NCCL_DEBUG=WARN" ;;
    P) run P $J1011 off 0 jeans 10 AMD_SERIALIZE_COPY=3 ;;
  esac
done
