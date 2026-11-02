#!/bin/bash -l
#SBATCH --job-name=luna_onezone
#SBATCH --partition=gpudev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=onezone-%j.out
#SBATCH --error=onezone-%j.err

set -eo pipefail
TABLE_PATH=${1:?Usage: sbatch submit/examples/nuclear_onezone_raven.sbatch /path/to/helm_table.dat}
module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
cd "${SLURM_SUBMIT_DIR}"
export JAX_PLATFORMS=cuda
export XLA_FLAGS=
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python -c 'import jax; assert all(d.platform == "gpu" for d in jax.devices()), jax.devices()'
python -m pytest -q tests/test_aprox13.py
python -u tools/check_nuclear_burn.py "$TABLE_PATH" --screening screen5 --neutrinos sneut5 --scipy --report "onezone-${SLURM_JOB_ID}.json"
