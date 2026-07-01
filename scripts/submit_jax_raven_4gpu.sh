#!/bin/bash -l
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
#SBATCH -D ./
#SBATCH -J jax_hydro  # Change the job name as needed
#SBATCH --partition=gpu1 
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:4  # 4 A100s (shard_map)
#SBATCH --cpus-per-task=72
#SBATCH --mem=125000
#SBATCH --time=01:00:00
#SBATCH --mail-type=BEGIN,FAIL
#SBATCH --mail-user=uli@mpa-garching.mpg.de

# Load required modules for the CUDA environment
module purge
module load gcc/14 openmpi cuda/13  # Loading the necessary modules for CUDA

# Activate the virtual environment where JAX with CUDA is installed
source ~/jax-env/bin/activate
# Set the JAX platform to CUDA (for Nvidia GPUs)
export JAX_PLATFORM_NAME=cuda  # This tells JAX to use CUDA for GPU acceleration

# Run your Python script using srun to execute on the allocated GPU
srun python main.py  # Executes the Python script that uses JAX
