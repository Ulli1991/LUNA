#!/bin/bash -l
#SBATCH --job-name=luna_he_det
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=detonation-%j.out
#SBATCH --error=detonation-%j.err

set -eo pipefail
TABLE_PATH=${1:?Usage: sbatch --partition=GPU_PARTITION submit/examples/detonation_raven.sbatch /path/to/helm_table.dat}
module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
cd "${SLURM_SUBMIT_DIR}"
export JAX_PLATFORMS=cuda
export XLA_FLAGS=
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python -c 'import jax, scipy; assert all(d.platform == "gpu" for d in jax.devices()), jax.devices(); print("JAX", jax.__version__, "SciPy", scipy.__version__, jax.devices(), flush=True)'
python -u tools/check_detonation_hydro.py "$TABLE_PATH" \
    "${LUNA_REFERENCE_SOLUTIONS:-../luna-reference-solutions}/luna/wd_validation/znd_he_tail_cpu.json" --grids 128 256 512 \
    --report "detonation-he-${SLURM_JOB_ID}.json"
python tools/check_detonation_front.py "detonation-he-${SLURM_JOB_ID}.json" \
    --output "detonation-he-front-${SLURM_JOB_ID}.json"
