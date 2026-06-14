#!/bin/bash -l
#SBATCH -o slurm_logs/ad_figs.out.%j
#SBATCH -e slurm_logs/ad_figs.err.%j
#SBATCH -D ./
#SBATCH -J luna_adfigs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=60000
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
cd /raven/u/uli/jax-hydro

# Regenerate the 3 title-bearing AD README figures with the title-free style.
echo "=== A1: Sedov AD validation ==="
python -u -m scripts.autodiff.validate_sedov

echo "=== ic_inversion: driven turbulence (Nx=32) ==="
LUNA_TURB_NX=32 LUNA_TURB_NSTEPS=200 LUNA_TURB_NITER=80 \
  python -u -m scripts.validation.runs.turb_autodiff_run

echo "=== ic_inversion: radiative-cooling box (Nx=32) ==="
LUNA_AD_PROBLEM=cooling LUNA_AD_NX=32 LUNA_AD_NSTEPS=150 LUNA_AD_NITER=80 \
  python -u -m scripts.validation.runs.ic_inversion_run
