#!/bin/bash -l
#SBATCH -o slurm_logs/adjoint_test.out.%j
#SBATCH -e slurm_logs/adjoint_test.err.%j
#SBATCH -D ./
#SBATCH -J luna_adjoint
#SBATCH --partition=gpudev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=125000
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#SBATCH --time=00:15:00

module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
cd /raven/u/uli/jax-hydro
python -u -m tests.test_poisson_adjoint
