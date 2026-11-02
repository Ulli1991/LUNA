#!/bin/bash -l
#SBATCH --job-name=luna_wd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=stellar-%j.out
#SBATCH --error=stellar-%j.err

set -eo pipefail
TABLE_PATH=${1:?Usage: sbatch --partition=GPU_PARTITION submit/examples/stellar_explosion_raven.sbatch /path/to/helm_table.dat}
module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
cd "${SLURM_SUBMIT_DIR}"
export JAX_PLATFORMS=cuda
export XLA_FLAGS=
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
RUN_OUTPUT="${SLURM_SUBMIT_DIR}/stellar-explosion-${SLURM_JOB_ID}"
mkdir "$RUN_OUTPUT"
python - "$TABLE_PATH" "$RUN_OUTPUT/manifest.json" <<'PY'
import hashlib, json, os, platform, subprocess, sys
from pathlib import Path
import jax, scipy
assert all(d.platform == "gpu" for d in jax.devices()), jax.devices()
table_hash = hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest()
expected_hash = json.loads(Path("tests/data/detonation_cj_reference.json").read_text())["table_sha256"]
assert table_hash == expected_hash, "Helmholtz table differs from the validated table"
manifest = dict(jax=jax.__version__, scipy=scipy.__version__, python=platform.python_version(),
                devices=list(map(str, jax.devices())),
                git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                git_status=subprocess.check_output(["git", "status", "--porcelain"], text=True),
                table_sha256=table_hash,
                requested_stop_time_s=float(os.environ.get("STOP_TIME", "0.1")),
                restart=os.environ.get("RESTART_CHECKPOINT"))
Path(sys.argv[2]).write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest), flush=True)
PY
restart_args=()
if [[ -n ${RESTART_CHECKPOINT:-} ]]; then
    : "${INITIAL_CHECKPOINT:?Set INITIAL_CHECKPOINT to the original step-zero checkpoint}"
    restart_args=(--restart "$RESTART_CHECKPOINT")
else
    INITIAL_CHECKPOINT="$RUN_OUTPUT/state_00000000.h5"
fi
run_status=0
python -u main.py stellar "$TABLE_PATH" \
    --config "${CONFIG:-scripts/examples/stellar/white_dwarf_explosion.json}" \
    --stop-time "${STOP_TIME:-0.1}" --dt-max "${DT_MAX:-1e-4}" \
    --checkpoint-every 25 --max-steps 200000 --output "$RUN_OUTPUT" \
    "${restart_args[@]}" || run_status=$?
shopt -s nullglob
checkpoints=("$RUN_OUTPUT"/state_*.h5)
if [[ -n ${RESTART_CHECKPOINT:-} ]]; then
    checkpoints+=("$RESTART_CHECKPOINT")
fi
if (( ${#checkpoints[@]} )); then
    python tools/analyze_stellar.py "${checkpoints[@]}" --initial "$INITIAL_CHECKPOINT" \
        --report "$RUN_OUTPUT/history.json" --profiles "$RUN_OUTPUT/profiles.h5"
fi
exit "$run_status"
