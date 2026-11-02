#!/bin/bash -l
# Build the tarball that prep_shm_env.sh broadcasts to /dev/shm on every
# compute node (venv + LUNA repo). Rerun after changing the repo or venv.
# Export REPO (your LUNA checkout) and LUNA_SCRATCH (your scratch directory).
set -eu
LUNA_SCRATCH=${LUNA_SCRATCH:?set LUNA_SCRATCH to your scratch directory}
OUT=$LUNA_SCRATCH/shm_env
mkdir -p "$OUT"
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
mkdir "$STAGE/luna_env"
cp -a $LUNA_SCRATCH/jax_env_cuda "$STAGE/luna_env/"
# the interpreter + stdlib too: at 4096 ranks the module tree on GPFS is
# still an import storm (6m20s startup with only venv+repo local)
module load Python/3.13.5
cp -a "$EBROOTPYTHON" "$STAGE/luna_env/python"
SRC=${REPO:?set REPO to your LUNA checkout}
rsync -a --exclude=.git --exclude='*.log' "$SRC" "$STAGE/luna_env/"
# .git stays behind, so record the exact code here: output headers read it (code_revision)
REV=$(git -C "$SRC" rev-parse HEAD)
if [ -n "$(git -C "$SRC" status --porcelain --untracked-files=no)" ]; then REV="$REV-dirty"; fi
echo "$REV" > "$STAGE/luna_env/LUNA/CODE_REVISION"
tar -cf "$OUT/luna_env.tar" -C "$STAGE" luna_env
ls -lh "$OUT/luna_env.tar"
