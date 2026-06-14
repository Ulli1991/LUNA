#!/bin/bash -l
#SBATCH -o slurm_logs/poisson_acc.out.%j
#SBATCH -e slurm_logs/poisson_acc.err.%j
#SBATCH -D ./
#SBATCH -J luna_poisson_acc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=60000
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00

module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export JAX_PLATFORM_NAME=cuda
export LUNA_PA_NX=${LUNA_PA_NX:-64}
srun python -u -m scripts.validation.runs.poisson_accuracy_run
