#!/bin/bash -l
#SBATCH -o slurm_logs/pm_force_test.out.%j
#SBATCH -e slurm_logs/pm_force_test.err.%j
#SBATCH -J pm_force
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40000
#SBATCH --time=00:20:00
module purge; module load gcc/13 cuda/12
source ~/jax-env/bin/activate
export JAX_PLATFORM_NAME=cuda
cd /raven/u/uli/jax-hydro
python -u -m scripts.validation.runs.pm_force_test
