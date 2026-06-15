#!/bin/bash -l
#SBATCH -o slurm_logs/regen_figs.out.%j
#SBATCH -e slurm_logs/regen_figs.err.%j
#SBATCH -D ./
#SBATCH -J luna_regen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=115000
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
cd /raven/u/uli/jax-hydro

# Regenerate every hydro/wind README figure from EXISTING snapshots with the
# now-consistent figstyle (style only — layouts unchanged). Each script is
# isolated with `|| true` so one failure doesn't abort the rest.
run() { echo "=== $* ==="; python -u -m "$@" || echo "  [FAILED] $*"; }

run scripts.validation.plots.shock_panels                 # sod / shu-osher / woodward-colella
run scripts.validation.plots.soundwave_convergence
run scripts.validation.plots.soundwave_standing_convergence
run scripts.validation.plots.sedov3d_plot                 # sedov3d + sedov3d_radial
run scripts.validation.plots.square_advection
run scripts.validation.plots.gresho_grid                  # gresho_vphi / density
run scripts.validation.plots.cc85_wind_grid               # m82_wind_* + cc85_wind_*
run scripts.validation.plots.turbulence_panel             # large (1024^3) reads
run scripts.validation.plots.kh_grid_plot                 # KH grid (READS snapshots; kh_grid is the RUNNER — do not use)
run scripts.validation.plots.kh_8192_compare              # large (8192^2) read

echo "=== DONE: regenerated figures ==="
ls -la figures/*.png | tail -40
