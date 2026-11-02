#!/bin/bash -l
#SBATCH -J adjcheck
#SBATCH -o slurm_logs/adj_check.%j.out
#SBATCH -e slurm_logs/adj_check.%j.err
#SBATCH --partition=apu --account=<your_account> --constraint=apu
#SBATCH --nodes=2 --ntasks-per-node=2 --cpus-per-task=24 --gres=gpu:2 --mem=200000
#SBATCH --time=00:40:00
# Correctness across nodes: the gradient, J^T w (and J v, hydro) on the whole
# mesh against one GPU of the same job, the transpose identity (hydro) and the
# mass identity. One "CHECK PASS|FAIL" line per problem.
#   sbatch --nodes=2 check.sbatch            MESH=2x2x1 (4 GPUs)
#   sbatch --nodes=4 --export=ALL,MESH=4x2x1 check.sbatch   (8 GPUs)
source submit/examples/autodiff_multinode/env.sh
O=$ADJ_OUT
NX=${NX:-128}; STEPS=${STEPS:-20}; MESH=${MESH:-2x2x1}; MESH=${MESH//x/,}
for prob in ${PROBS:-hydro jeans cooling}; do
    A="problem=$prob nx=$NX n_steps=$STEPS"
    R=$O/ref_${prob}_${NX}_${STEPS}_$SLURM_JOB_ID.npz
    echo "#### $prob $(date +%T)"
    adj_srun -N1 -n1 --gres=gpu:1 python tools/adj_multinode.py ref $A mesh=1,1,1 out=$R
    adj_srun python tools/adj_multinode.py check $A mesh=$MESH ref=$R
    echo "#### end $prob rc=$?"
done
