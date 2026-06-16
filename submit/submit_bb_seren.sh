#!/bin/bash -l
#SBATCH -o slurm_logs/bb_seren.out.%j
#SBATCH -e slurm_logs/bb_seren.err.%j
#SBATCH -D ./
#SBATCH -J bbseren
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120000
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
module purge; module load gcc/13 cuda/12
source ~/jax-env/bin/activate
cd /raven/u/uli/jax-hydro
export LUNA_BB_NX=${LUNA_BB_NX:-256}
# LUNA_BB_SINKS, LUNA_BB_TEND_MYR (default 0.06) inherited from --export
python -u -m scripts.validation.runs.boss_bodenheimer_iso_run
