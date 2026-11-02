#!/bin/bash -l
#SBATCH -J sil_train
#SBATCH -o slurm_logs/sil_train.%j.out
#SBATCH -e slurm_logs/sil_train.%j.err
#SBATCH --partition=apu --account=<your_account> --constraint=apu
#SBATCH --nodes=4 --ntasks-per-node=2 --cpus-per-task=24 --gres=gpu:2 --mem=200000
#SBATCH --time=06:00:00
# The 3D solver-in-the-loop example on the 256^3 coarse grid (tools/sil3d.py):
# SiL training through unrolled rollouts (revolve), the same network trained
# one-step supervised (offline), both evaluated beyond the training window.
# Submit from the repo root after ref.sbatch has finished.
source submit/examples/autodiff_multinode/env.sh
REF=${REF:-$ADJ_OUT/sil3d/ref_m5_s7_1024}
RUN=${RUN:-$ADJ_OUT/sil3d/run_m5_s7}
MESH=${MESH:-2,2,2}
TRAIN=${TRAIN:-1900:2900}
mkdir -p $RUN
if [ ! -f $RUN/sil/weights.npz ]; then
  adj_srun python tools/sil3d.py train ref=$REF out=$RUN/sil mesh=$MESH train=$TRAIN \
      unroll=${UNROLL:-8,32,128} iters=${ITERS:-200,150,100} every=4 lr=${LR:-1e-3} snaps=16 log_every=10
fi
if [ ! -f $RUN/offline/weights.npz ]; then
  adj_srun python tools/sil3d.py offline ref=$REF out=$RUN/offline mesh=$MESH train=$TRAIN \
      iters=${OFF_ITERS:-1500} lr=${LR:-1e-3} log_every=50
fi
adj_srun python tools/sil3d.py eval ref=$REF out=$RUN/eval_n2900.pkl mesh=$MESH n0=2900 \
    steps=${EVAL_STEPS:-4640} every=160 weights=$RUN/sil/weights.npz,$RUN/offline/weights.npz
for m in stats time slices; do
  python tools/sil3d_figs.py $m $RUN/eval_n2900.pkl $RUN/fig_$m.png t_from=1.0 k_train=128
done
python tools/sil3d_figs.py hist $RUN/sil/hist.npy,$RUN/offline/hist.npy $RUN/fig_hist.png
