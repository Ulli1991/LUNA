#!/bin/bash -l
#SBATCH --job-name=luna_helmholtz
#SBATCH --partition=gpudev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=helmholtz-%j.out
#SBATCH --error=helmholtz-%j.err

set -eo pipefail
TABLE_PATH=${1:?Usage: sbatch submit/examples/helmholtz_raven.sbatch /path/to/helm_table.dat}
module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
cd "${SLURM_SUBMIT_DIR}"
export JAX_PLATFORMS=cuda
export XLA_FLAGS=
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python -u tools/check_helmholtz.py "$TABLE_PATH" --sweep --require-gpu --report "helmholtz-${SLURM_JOB_ID}.json"
python -m pytest -q tests/test_helmholtz.py
