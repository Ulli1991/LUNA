#!/bin/bash -l
#SBATCH -J adjcool
#SBATCH -o slurm_logs/adj_cooldiag.%j.out
#SBATCH -e slurm_logs/adj_cooldiag.%j.err
#SBATCH --partition=apudev --account=<your_account> --constraint=apu
#SBATCH --nodes=1 --ntasks-per-node=2 --cpus-per-task=24 --gres=gpu:2 --mem=200000
#SBATCH --time=00:15:00
# The cooling reverse-mode crash, bisected on ONE GPU (it needs no second one):
#   sbatch --export=ALL,CASES="s60 s120 s120copy s120hlb" cooldiag.sbatch
source submit/examples/autodiff_multinode/env.sh
unset AMD_SERIALIZE_COPY          # the cases set it themselves
NX=${NX:-128}
v() {  # v <label> <timeout> <args...>
    l=$1; t=$2; shift 2
    echo "#### $l: $* HLB=${HIP_LAUNCH_BLOCKING:-0} SDMA=${HSA_ENABLE_SDMA:-} QUEUES=${GPU_MAX_HW_QUEUES:-} SCOPY=${AMD_SERIALIZE_COPY:-} $(date +%T)"
    timeout $t srun -n1 --gres=gpu:1 python tools/adj_multinode.py probe problem=cooling nx=$NX ops=fwd,vjp calls=${CALLS:-3} "$@" 2>&1 | grep -E "PROBE|done|start|Error|rror:" | tail -12
    echo "#### end $l rc=${PIPESTATUS[0]}"
}
for c in ${CASES:-base base256 hlb}; do
  case $c in
    base)   v base 200 n_steps=20 ;;
    ou0)    v ou0 200 n_steps=20 ou=0 ;;
    nocool) v nocool 200 n_steps=20 cooling_enabled=False ;;
    nsub1)  v nsub1 200 n_steps=20 cooling_n_sub=1 ;;
    steps5) v steps5 200 n_steps=5 ;;
    steps1) v steps1 200 n_steps=1 ;;
    base256) NX=256 v base256 300 n_steps=20 ;;
    hlb)    HIP_LAUNCH_BLOCKING=1 v hlb 300 n_steps=20 ;;
    s120)   v s120 200 n_steps=120 ;;
    s60)    v s60 200 n_steps=60 ;;
    s120n4) v s120n4 200 n_steps=120 cooling_n_sub=4 ;;
    s120ou0) v s120ou0 200 n_steps=120 ou=0 ;;
    s120hlb) HIP_LAUNCH_BLOCKING=1 v s120hlb 300 n_steps=120 ;;
    s120sdma) HSA_ENABLE_SDMA=0 v s120sdma 200 n_steps=120 ;;
    s120q1) GPU_MAX_HW_QUEUES=1 v s120q1 200 n_steps=120 ;;
    s120copy) AMD_SERIALIZE_COPY=3 v s120copy 200 n_steps=120 ;;
    s20copy) AMD_SERIALIZE_COPY=3 v s20copy 200 n_steps=20 ;;
    flat20) v flat20 200 n_steps=20 state=flat ;;
    s120copy1) AMD_SERIALIZE_COPY=1 v s120copy1 200 n_steps=120 ;;
    s120copy2) AMD_SERIALIZE_COPY=2 v s120copy2 200 n_steps=120 ;;
  esac
done
