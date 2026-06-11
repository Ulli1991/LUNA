#!/bin/bash -l
#SBATCH -o slurm_logs/bb_per_min.out.%j
#SBATCH -e slurm_logs/bb_per_min.err.%j
#SBATCH -D ./
#SBATCH -J luna_bb_per_min
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=60000
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
cd /raven/u/uli/jax-hydro

export LUNA_BB_NX=${LUNA_BB_NX:-256}
export LUNA_BB_TEND=${LUNA_BB_TEND:-2.0}

python -u -m scripts.validation.runs.boss_bodenheimer_periodic_minmod_run
