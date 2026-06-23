#!/bin/bash -l
#SBATCH -o /ptmp/uli/jax-hydro/snapshots/bench_fmg.out.%j
#SBATCH -e /ptmp/uli/jax-hydro/snapshots/bench_fmg.err.%j
#SBATCH -D /raven/u/uli/jax-hydro
#SBATCH -J bench_fmg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=240000
#SBATCH --time=00:15:00
module purge; module load gcc/13 cuda/12; source ~/jax-env/bin/activate
export JAX_PLATFORM_NAME=cuda XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONUNBUFFERED=1
export BENCH_NX=${BENCH_NX:-512}
cd /raven/u/uli/jax-hydro
python -u -m gravity.bench_fmg
