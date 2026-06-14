#!/bin/bash -l
#SBATCH -o slurm_logs/m82ad_multi.out.%j
#SBATCH -e slurm_logs/m82ad_multi.err.%j
#SBATCH -D ./
#SBATCH -J luna_m82ad_multi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --mem=500000
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
module purge
module load gcc/13 cuda/12
source ~/jax-env/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export JAX_PLATFORM_NAME=cuda
export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_deterministic_ops=true"
NODES=($(scontrol show hostname $SLURM_JOB_NODELIST))
HEAD_NODE=${NODES[0]}
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)
export JAX_COORDINATOR_ADDRESS="${HEAD_NODE_IP}:12345"
echo "Coordinator: $JAX_COORDINATOR_ADDRESS, nodes=${SLURM_JOB_NUM_NODES}"
srun python -u -m scripts.validation.runs.m82_autodiff_run
