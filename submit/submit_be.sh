#!/bin/bash -l
#SBATCH -o slurm_logs/be.out.%j
#SBATCH -e slurm_logs/be.err.%j
#SBATCH -D ./
#SBATCH -J luna_be
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

export LUNA_BE_NX=${LUNA_BE_NX:-64}
export LUNA_BE_XI_T=${LUNA_BE_XI_T:-5.0}
export LUNA_BE_TEND=${LUNA_BE_TEND:-3.0}

srun python -u -m scripts.validation.runs.bonnor_ebert_run
