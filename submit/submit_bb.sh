#!/bin/bash -l
#SBATCH -o slurm_logs/bb.out.%j
#SBATCH -e slurm_logs/bb.err.%j
#SBATCH -D ./
#SBATCH -J luna_bb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=60000
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export JAX_PLATFORM_NAME=cuda

export LUNA_BB_NX=${LUNA_BB_NX:-64}
export LUNA_BB_TEND=${LUNA_BB_TEND:-2.0}

srun python -u -m scripts.validation.runs.boss_bodenheimer_run
