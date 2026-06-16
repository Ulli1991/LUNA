#!/bin/bash -l
#SBATCH -o slurm_logs/scal_luna_128_1.out.%j
#SBATCH -e slurm_logs/scal_luna_128_1.err.%j
#SBATCH -D ./
#SBATCH -J sL_128_1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=30000
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
module purge; module load gcc/13 cuda/12
source ~/jax-env/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export JAX_PLATFORM_NAME=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LUNA_SEDOV_NX=128
export LUNA_USE_SHARDMAP=0
export LUNA_MESH_SHAPE=1,1,1
export LUNA_SEDOV_NSNAPS=1
export LUNA_SEDOV_OUT=/ptmp/uli/jax-hydro/snapshots/scal_luna_128_1
mkdir -p "$LUNA_SEDOV_OUT"
srun python -u -m scripts.validation.runs.sedov3d_run
