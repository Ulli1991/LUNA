#!/bin/bash -l
#SBATCH -o slurm_logs/kh_corrector.out.%j
#SBATCH -e slurm_logs/kh_corrector.err.%j
#SBATCH -D ./
#SBATCH -J luna_kh_corr
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
export LUNA_KHC_NX=${LUNA_KHC_NX:-32}
export LUNA_KHC_NSTEPS=${LUNA_KHC_NSTEPS:-200}
export LUNA_KHC_NITER=${LUNA_KHC_NITER:-80}
srun python -u -m scripts.validation.runs.kh_corrector_run
