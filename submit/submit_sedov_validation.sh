#!/bin/bash -l
#SBATCH -o slurm_logs/sedov_val.out.%j
#SBATCH -e slurm_logs/sedov_val.err.%j
#SBATCH -D ./
#SBATCH -J luna_sedov_val
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=60000
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:15:00

module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export JAX_PLATFORM_NAME=cuda
srun python -u -m scripts.autodiff.validate_sedov
