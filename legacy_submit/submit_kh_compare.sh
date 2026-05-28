#!/bin/bash -l
#SBATCH -o ./kh_cmp.out.%j
#SBATCH -e ./kh_cmp.err.%j
#SBATCH -D ./
#SBATCH -J kh_cmp
#SBATCH --partition=gpudev
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#SBATCH --time=00:15:00

module purge
module load gcc/13
source ~/jax-env/bin/activate

python -u -m scripts.validation.plots.kh_8192_compare
