#!/bin/bash -l
#SBATCH -p apudev
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j 
#SBATCH -D ./
#SBATCH -J sod_3d
#SBATCH --ntasks=2                 # TWO MPI TASKS
#SBATCH --gres=gpu:2               # TWO GPUS 
#SBATCH --gpus-per-task=1          # ONE GPU PER MPI RANK
#SBATCH --cpus-per-task=12         # Divide the 24 CPUs between tasks
#SBATCH --mem=110000  
#SBATCH --time=00:15:00 
#SBATCH --mail-type=BEGIN,FAIL
#SBATCH --mail-user=uli@mpa-garching.mpg.de 

module purge
module load gcc/14 openmpi/5.0 rocm/7.0

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
source ~/venv_jax_rocm7.0/bin/activate

# --- FIX: make sure mpi4jax / mpi4py see the right MPI ---
export LD_LIBRARY_PATH=${OPENMPI_HOME}/lib:$LD_LIBRARY_PATH
export PATH=${OPENMPI_HOME}/bin:$PATH
export MPICC=${OPENMPI_HOME}/bin/mpicc   # helps mpi4py + mpi4jax

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export JAX_PLATFORM_NAME=rocm

# IMPORTANT: run with srun so SLURM launches the MPI ranks:
srun python main_multigpu_mpi.py
