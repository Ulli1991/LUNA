#!/bin/bash -l
# LUNA multi-GPU on viper — 2× MI300A (ROCm), single node.
# JAX 1-GPU-per-process model: 2 tasks, 2 GPUs, coordinated via JAX distributed.
#SBATCH -o slurm_logs/luna_2gpu.out.%j
#SBATCH -e slurm_logs/luna_2gpu.err.%j
#SBATCH -D ./
#SBATCH -J luna_2gpu
#SBATCH --partition=apudev
#SBATCH --account=<your_account>
#SBATCH --constraint=apu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2        # ONE task per GPU
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2               # both MI300A on the node
#SBATCH --mem=200000
#SBATCH --time=00:15:00

module purge
module load rocm/7.2 gcc/14
source ~/jax_env2/bin/activate

export JAX_PLATFORMS=rocm
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# JAX distributed coordinator (single node -> use this node's IP)
NODES=($(scontrol show hostname $SLURM_JOB_NODELIST))
HEAD_NODE=${NODES[0]}
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)
export JAX_COORDINATOR_ADDRESS="${HEAD_NODE_IP}:12345"

echo "=== node: $HEAD_NODE ($HEAD_NODE_IP)  ntasks=$SLURM_NTASKS  coord=$JAX_COORDINATOR_ADDRESS ==="

# viper's SLURM exposes BOTH GPUs to BOTH tasks (ROCR_VISIBLE_DEVICES=0,1). Do
# NOT restrict visibility per task — the ROCm backend fails under JAX distributed
# if a rank only sees its non-zero global id. Instead LUNA pins each rank to its
# GPU via local_device_ids=[SLURM_LOCALID] inside initialize_jax_distributed().
srun bash -c '
  echo "rank ${SLURM_PROCID} localid ${SLURM_LOCALID} host $(hostname)"
  python main.py
'
echo "=== done ==="
