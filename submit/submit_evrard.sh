#!/bin/bash -l
#SBATCH -o slurm_logs/evrard.out.%j
#SBATCH -e slurm_logs/evrard.err.%j
#SBATCH -D ./
#SBATCH -J luna_evrard
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

export LUNA_EVRARD_NX=${LUNA_EVRARD_NX:-64}
export LUNA_EVRARD_TEND=${LUNA_EVRARD_TEND:-1.5}

srun python -u -m scripts.validation.runs.evrard_collapse_run
