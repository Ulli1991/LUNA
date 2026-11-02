#!/bin/bash -l
#SBATCH -J adjicinv
#SBATCH -o slurm_logs/adj_icinv.%j.out
#SBATCH -e slurm_logs/adj_icinv.%j.err
#SBATCH --partition=apu --account=<your_account> --constraint=apu
#SBATCH --nodes=4 --ntasks-per-node=2 --cpus-per-task=24 --gres=gpu:2 --mem=200000
#SBATCH --time=00:45:00
# The 256^3 gravity IC inversion (the one that hung at walltime) on 8 GPUs
# over 4 nodes; the same script with --nodes=1 MESH=2x1x1 is the baseline.
#   sbatch icinv.sbatch                                     8 GPUs, 4x2x1
#   sbatch --nodes=1 --export=ALL,MESH=2x1x1 icinv.sbatch    2 GPUs
source submit/examples/autodiff_multinode/env.sh
MESH=${MESH:-4x2x1}; MESH=${MESH//x/,}
echo "IC INVERSION ${PROB:-jeans} nx=${NX:-256} n_steps=${STEPS:-120} n_iter=${NITER:-30} mesh=$MESH tasks=$SLURM_NTASKS"
adj_srun python main.py scripts/examples/autodiff/ic_inversion_run.py problem=${PROB:-jeans} \
    nx=${NX:-256} n_steps=${STEPS:-120} n_iter=${NITER:-30} LIMITER=0 \
    MESH_SHAPE=$MESH output_dir=$ADJ_OUT/icinv_$SLURM_JOB_ID
