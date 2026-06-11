#!/bin/bash -l
#SBATCH -o slurm_logs/mv_to_ptmp.out.%j
#SBATCH -e slurm_logs/mv_to_ptmp.err.%j
#SBATCH -D ./
#SBATCH -J mv_snaps
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000
#SBATCH --time=24:00:00

set -e
mkdir -p /ptmp/uli/jax-hydro/snapshots /ptmp/uli/jax-hydro/checkpoints

# Move each sub-directory individually. Skip:
#  * Anything that is already a symlink (live run is writing through it).
#  * Anything where the destination already exists (avoid overwrite).
# After each successful move, replace the source with a symlink so
# subsequent jobs writing to that path still resolve correctly.
for d in /raven/u/uli/jax-hydro/snapshots/* /raven/u/uli/jax-hydro/checkpoints/*; do
  if [ -L "$d" ]; then
    echo "SKIP (symlink, live run): $d"
    continue
  fi
  if [ ! -d "$d" ]; then
    continue
  fi
  name=$(basename "$d")
  parent=$(basename "$(dirname "$d")")   # snapshots or checkpoints
  dst="/ptmp/uli/jax-hydro/$parent/$name"
  if [ -e "$dst" ]; then
    echo "SKIP (destination exists): $dst"
    continue
  fi
  echo "Moving $d -> $dst"
  mv "$d" "$dst"
  ln -s "$dst" "$d"
done

echo "ALL DONE"
