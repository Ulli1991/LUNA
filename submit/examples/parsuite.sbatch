#!/bin/bash -l
#SBATCH -p apudev --account=<your_account> --constraint=apu --gres=gpu:2 -N1 -n1 --cpus-per-task=48 --mem=200000 --time=00:15:00
#SBATCH -o parsuite_%x.%j.out
# Full suite, CPU backend, in parallel on one apudev node:
#   sbatch -J <name> --export=ALL,ROOT=<tree>,OUT=<scratch dir> submit/examples/parsuite.sbatch
# A tree older than tools/parsuite.py: add RUNNER=<a checkout>/tools/parsuite.py.
module purge; module load gcc/14 rocm/7.2 openmpi
source ${VENV:?set VENV to a venv with jax 0.10.0 and AMUSE}/bin/activate
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export LUNA_DATA=${LUNA_DATA:?set LUNA_DATA to the data directory}
OUT=${OUT:?set OUT to a scratch directory}
export LUNA_TEST_SCRATCH=$OUT/scratch_${SLURM_JOB_NAME}
mkdir -p $LUNA_TEST_SCRATCH
python ${RUNNER:-$ROOT/tools/parsuite.py} $ROOT $OUT/parsuite_${SLURM_JOB_NAME}.json 8
