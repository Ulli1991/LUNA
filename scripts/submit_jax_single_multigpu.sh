#!/bin/bash -l
# JAX Sharding - Single-node multi-GPU on Raven A100
#SBATCH -o ./job_single4gpu.out.%j
#SBATCH -e ./job_single4gpu.err.%j
#SBATCH -D ./
#SBATCH -J jax_4gpu
#
# ===================================================================
# Single-node 4-GPU configuration
# ===================================================================
#SBATCH --ntasks=1              # Single task sees all GPUs
#SBATCH --cpus-per-task=72      # Full node
#SBATCH --mem=500000            # Full node memory
#SBATCH --constraint="gpu"      # A100 nodes
#SBATCH --gres=gpu:4            # All 4 GPUs
#
#SBATCH --time=01:00:00
#SBATCH --mail-type=BEGIN,FAIL
#SBATCH --mail-user=uli@mpa-garching.mpg.de

# Load modules
module purge
module load gcc/13 cuda/12

# Activate JAX environment
source ~/venv_jax_cuda/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# CRITICAL: Ensure all 4 GPUs are visible to JAX
# SLURM might restrict visibility, so explicitly set it
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Force CUDA platform (not ROCm!)
export JAX_PLATFORM_NAME=cuda

echo "==================================================================="
echo "JAX Single-Node Multi-GPU"
echo "==================================================================="
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "GPUs: ${SLURM_GPUS_ON_NODE:-unknown}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "JAX_PLATFORM_NAME: ${JAX_PLATFORM_NAME}"
echo "==================================================================="
# Verify GPUs are visible to nvidia-smi
nvidia-smi --list-gpus
echo "==================================================================="

# Single node - JAX automatically sees all 4 GPUs
# Don't use srun for single-process multi-GPU - it restricts GPU visibility
python main_multi.py

echo "Simulation completed!"
