#!/bin/bash -l
#SBATCH -o slurm_logs/ou_inverse.out.%j
#SBATCH -e slurm_logs/ou_inverse.err.%j
#SBATCH -D ./
#SBATCH -J luna_ou_inv
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
export XLA_FLAGS="--xla_gpu_autotune_level=1"
export LUNA_TURB_NX=${LUNA_TURB_NX:-32}
export LUNA_TURB_NSTEPS=${LUNA_TURB_NSTEPS:-150}
export LUNA_TURB_NITER=${LUNA_TURB_NITER:-60}
srun python -u -m scripts.validation.runs.turb_autodiff_run
