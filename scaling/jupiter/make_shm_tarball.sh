#!/bin/bash -l
# Build the tarball that prep_shm_env.sh broadcasts to /dev/shm on every
# compute node (venv + LUNA repo). Rerun after changing the repo or venv.
set -eu
OUT=/e/scratch/luna/uli/shm_env
mkdir -p "$OUT"
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
mkdir "$STAGE/luna_env"
cp -a /e/scratch/luna/uli/jax_env_cuda "$STAGE/luna_env/"
# the interpreter + stdlib too: at 4096 ranks the module tree on GPFS is
# still an import storm (6m20s startup with only venv+repo local)
module load Python/3.13.5
cp -a "$EBROOTPYTHON" "$STAGE/luna_env/python"
rsync -a --exclude=.git --exclude='*.log' \
    /e/home/jusers/steinwandel2/jupiter/scaling_luna/LUNA "$STAGE/luna_env/"
tar -cf "$OUT/luna_env.tar" -C "$STAGE" luna_env
ls -lh "$OUT/luna_env.tar"
