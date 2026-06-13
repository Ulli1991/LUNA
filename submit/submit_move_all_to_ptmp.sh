#!/bin/bash -l
#SBATCH -o slurm_logs/mv_all_ptmp.out.%j
#SBATCH -e slurm_logs/mv_all_ptmp.err.%j
#SBATCH -D ./
#SBATCH -J mv_all_ptmp
#SBATCH --partition=interactive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000
#SBATCH --time=01:59:00

# Move ALL simulation output dirs from the repo (home quota) to
# /ptmp/uli/jax-hydro and replace each with a top-level symlink, so every
# relative output_dir="snapshots/..." resolves to /ptmp transparently.
set -e
REPO=/raven/u/uli/jax-hydro
DEST=/ptmp/uli/jax-hydro
mkdir -p "$DEST"

for name in snapshots checkpoints snapshots_kh snapshots_kh_multi snapshots_sedov_sn snapshots_sod reference; do
  src="$REPO/$name"
  dst="$DEST/$name"
  if [ -L "$src" ]; then
    echo "SKIP (already symlink): $src"
    continue
  fi
  if [ ! -d "$src" ]; then
    echo "SKIP (missing): $src"
    continue
  fi
  mkdir -p "$dst"
  echo "Moving contents of $src -> $dst"
  # Move entries one by one so a partially-migrated dir (earlier mover run
  # left per-subdir symlinks) is handled: real dirs move, symlinks resolve
  # into $DEST already and are just removed.
  for entry in "$src"/* "$src"/.[!.]*; do
    [ -e "$entry" ] || [ -L "$entry" ] || continue
    base=$(basename "$entry")
    if [ -L "$entry" ]; then
      echo "  drop old symlink: $base"
      rm "$entry"
      continue
    fi
    if [ -e "$dst/$base" ]; then
      echo "  SKIP (exists at dest): $base"
      continue
    fi
    mv "$entry" "$dst/$base"
  done
  rmdir "$src"
  ln -s "$dst" "$src"
  echo "  done: $src -> $dst"
done

echo "ALL DONE"
df -h /raven/u | tail -1
