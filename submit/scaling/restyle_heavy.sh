#!/bin/bash -l
#SBATCH -o slurm_logs/restyle_heavy.out.%j
#SBATCH -e slurm_logs/restyle_heavy.err.%j
#SBATCH -D ./
#SBATCH -J restyle_heavy
#SBATCH --partition=gpudev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --time=00:15:00
module purge; module load gcc/13
source ~/jax-env/bin/activate
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
cd /raven/u/uli/jax-hydro
run() { echo "=== $1 ==="; if python -u -m "$1" >/tmp/rh_$1.log 2>&1; then echo "  OK"; else echo "  FAILED:"; tail -5 /tmp/rh_$1.log; fi; }
run scripts.validation.plots.sedov3d_plot      # sedov3d + radial, now incl 1024^3
run scripts.validation.plots.kh_grid_plot      # KH 4x5 grid
run scripts.validation.plots.kh_8192_compare   # 8192^2 vs AthenaK
run scripts.validation.plots.turbulence_panel  # 1024^3 turbulence slice
echo "=== restyled ==="; ls -la figures/sedov3d.png figures/kh_grid.png figures/kh_8192_luna_vs_athenak.png figures/turbulence_panel.png | awk '{print $6,$7,$8,$9}'
