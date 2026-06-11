#!/bin/bash -l
#SBATCH -o slurm_logs/jeans_conv.out.%j
#SBATCH -e slurm_logs/jeans_conv.err.%j
#SBATCH -D ./
#SBATCH -J luna_jeans_conv
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=60000
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export JAX_PLATFORM_NAME=cuda

NX=${LUNA_JEANS_NX:-128}
export LUNA_JEANS_NX=$NX
export LUNA_JEANS_TEND=1.0
unset LUNA_USE_SHARDMAP LUNA_MESH_SHAPE
export LUNA_SELF_GRAVITY=1

srun python -u -m scripts.validation.runs.jeans_instability_run
