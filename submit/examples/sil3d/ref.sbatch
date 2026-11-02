#!/bin/bash -l
#SBATCH -J sil_ref
#SBATCH -o slurm_logs/sil_ref.%j.out
#SBATCH -e slurm_logs/sil_ref.%j.err
#SBATCH --partition=apu --account=<your_account> --constraint=apu
#SBATCH --nodes=4 --ntasks-per-node=2 --cpus-per-task=24 --gres=gpu:2 --mem=200000
#SBATCH --time=12:00:00
# The fine reference of the 3D solver-in-the-loop example (tools/sil3d.py
# ref): 1024^3 driven isothermal turbulence, pooled 4x to 256^3. Spin-up to
# ~2 t_turn, every coarse step stored over ~1 t_turn (the training window),
# then every 16th to ~8 t_turn (evaluation). Resubmit the same command after
# a timeout: it restarts from out/restart. fuse=0: the stages compiled apart,
# as the production loop runs them and as the published reference was made
# (before the hydro's optimization barrier the whole step as one program went
# wrong at 1024^3 on 2x2x2, docs/AUTODIFF.md). Submit from the repo root.
#   sbatch submit/examples/sil3d/ref.sbatch
#   sbatch --export=ALL,MACH=2,SEED=11,REC=... submit/examples/sil3d/ref.sbatch
source submit/examples/autodiff_multinode/env.sh
MACH=${MACH:-5}; SEED=${SEED:-7}
OUT=${OUT:-$ADJ_OUT/sil3d/ref_m${MACH}_s${SEED}_1024}
N_END=${N_END:-7600}
REC=${REC:-1900:2900:1,2900:7601:16}
adj_srun python tools/sil3d.py ref out=$OUT nx=1024 f=4 mach=$MACH seed=$SEED mesh=${MESH:-2,2,2} \
    acc_coef=${ACC:-2.55} vfac=2 n_end=$N_END rec=$REC diag_every=32 wall_h=11.5 fuse=0
