#!/bin/bash -l
#SBATCH -o slurm_logs/jeans_nograv.out.%j
#SBATCH -e slurm_logs/jeans_nograv.err.%j
#SBATCH -D ./
#SBATCH -J luna_jeans_nograv_1node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=60000
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00

module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export JAX_PLATFORM_NAME=cuda

export LUNA_JEANS_NX=64
export LUNA_JEANS_TEND=1.0
# Single GPU, no shard_map, no gravity — pure soundwave reference.
export LUNA_SELF_GRAVITY=0
unset LUNA_USE_SHARDMAP
unset LUNA_MESH_SHAPE

srun python -u -m scripts.validation.runs.jeans_instability_run
