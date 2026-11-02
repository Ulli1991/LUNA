#!/bin/bash -l
#SBATCH -J adjbench
#SBATCH -o slurm_logs/adj_bench.%j.out
#SBATCH -e slurm_logs/adj_bench.%j.err
#SBATCH --partition=apu --account=<your_account> --constraint=apu
#SBATCH --nodes=4 --ntasks-per-node=2 --cpus-per-task=24 --gres=gpu:2 --mem=200000
#SBATCH --time=00:50:00
# Forward / JVP / VJP at scale, FP64 minmod, fixed dt. Each RUNS entry
# (ops joined by +, :steps) is its own process: the peak-memory mark is not resettable.
#   sbatch bench.sbatch                                   512^3 on 8 GPUs, 4x2x1
#   sbatch --export=ALL,PROB=jeans,RUNS="fwd+grad:8" bench.sbatch
source submit/examples/autodiff_multinode/env.sh
NX=${NX:-512}; MESH=${MESH:-4x2x1}; MESH=${MESH//x/,}; PROB=${PROB:-hydro}
for r in ${RUNS:-fwd+jvp:8 fwd+grad:8 grad:32 grad:64}; do
    ops=${r%%:*}; ops=${ops//+/,}; steps=${r##*:}
    echo "#### $PROB nx=$NX steps=$steps ops=$ops mesh=$MESH $(date +%T)"
    adj_srun python tools/adj_multinode.py bench problem=$PROB nx=$NX \
        n_steps=$steps mesh=$MESH ops=$ops reps=${REPS:-2}
    echo "#### end rc=$?"
done
