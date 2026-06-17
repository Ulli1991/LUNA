#!/bin/bash -l
#SBATCH -o slurm_logs/kh_Nx1024.out.%j
#SBATCH -e slurm_logs/kh_Nx1024.err.%j
#SBATCH -D ./
#SBATCH -J kh_1024
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=120000
#SBATCH --time=00:30:00
module purge; module load gcc/13 cuda/12
source ~/jax-env/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export JAX_PLATFORM_NAME=cuda
export LUNA_KH_NX=1024
export LUNA_KH_OUT=snapshots/kh_Nx1024
srun python -u -m scripts.validation.runs.kh_run
