#!/bin/bash -l
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j 
#SBATCH -D ./
#SBATCH -J jax_hydro 
#SBATCH --ntasks=1  
#SBATCH --constraint="apu"  
#SBATCH --gres=gpu:1  
#SBATCH --cpus-per-task=24 
#SBATCH --mem=110000  
#SBATCH --time=08:00:00 
#SBATCH --mail-type=BEGIN,FAIL
#SBATCH --mail-user=uli@mpa-garching.mpg.de 

module purge  
module load gcc/14 openmpi/5.0 rocm/7.0  # Load required modules

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
source ~/venv_jax_rocm7.0/bin/activate  # Adjust the path to your virtualenv
export JAX_PLATFORM_NAME=rocm  # This makes JAX use the ROCm platform (GPU)
#export JAX_PLATFORM_NAME=cpu  # This makes JAX use the ROCm platform (GPU)

srun python main.py  # Executes the Python script that uses JAX

