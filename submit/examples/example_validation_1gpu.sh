#!/bin/bash -l
#SBATCH -o slurm_logs/sedov3d_128.out.%j
#SBATCH -e slurm_logs/sedov3d_128.err.%j
#SBATCH -D /raven/u/uli/jax-hydro
#SBATCH -J sedov3d_128
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=120000
#SBATCH --time=00:20:00
module purge; module load gcc/13 cuda/12; source ~/jax-env/bin/activate
export JAX_PLATFORM_NAME=cuda; export PYTHONUNBUFFERED=1
export LUNA_SEDOV_NX=128
cd /raven/u/uli/jax-hydro
python -u -m scripts.validation.runs.sedov3d_run
echo "done rc=$?"
