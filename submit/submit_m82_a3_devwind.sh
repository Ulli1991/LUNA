#!/bin/bash -l
#SBATCH -o slurm_logs/m82a3.out.%j
#SBATCH -e slurm_logs/m82a3.err.%j
#SBATCH -D ./
#SBATCH -J luna_m82a3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=480000
#SBATCH --gres=gpu:a100:4
#SBATCH --time=04:00:00
module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
export JAX_PLATFORM_NAME=cuda
unset JAX_COORDINATOR_ADDRESS
# developed M82 wind (50 Myr), forward-mode sensitivity, 4-GPU auto-shard
export LUNA_M82_NX=128
export LUNA_M82_TEND=${LUNA_M82_TEND:-10}
export LUNA_M82_NSTEPS=8000
export LUNA_M82_FIXEDDT=0
export LUNA_M82_DTMODE=diff
export LUNA_M82_GRADMODE=forward
export LUNA_M82_SELFGRAV=0
export LUNA_M82_SHARDMAP=0
export LUNA_M82_MESH=2x2x1
python -u -m scripts.validation.runs.m82_autodiff_run
