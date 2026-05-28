#!/bin/bash -l
#SBATCH -o ./turb_panel.out.%j
#SBATCH -e ./turb_panel.err.%j
#SBATCH -D ./
#SBATCH -J turb_panel
#SBATCH --partition=gpudev
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=60000
#SBATCH --time=00:05:00

module purge
module load gcc/13
source ~/jax-env/bin/activate

python -u -m scripts.validation.plots.turbulence_panel
