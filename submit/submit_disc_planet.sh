#!/bin/bash -l
#SBATCH -o slurm_logs/disc_planet.out.%j
#SBATCH -e slurm_logs/disc_planet.err.%j
#SBATCH -D ./
#SBATCH -J luna_disc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=60000
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
cd /raven/u/uli/jax-hydro

export LUNA_DP_NX=${LUNA_DP_NX:-512}
export LUNA_DP_ORBITS=${LUNA_DP_ORBITS:-30}
export LUNA_DP_CS=${LUNA_DP_CS:-0.05}
export LUNA_DP_Q=${LUNA_DP_Q:-1.0e-3}
export LUNA_DP_SINK=${LUNA_DP_SINK:-1}

python -u -m scripts.validation.runs.disc_planet_run
