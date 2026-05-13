#!/bin/bash -l
# JAX Multi-Node on MI300A (ROCm)
#SBATCH -o ./job_multi.out.%j
#SBATCH -e ./job_multi.err.%j
#SBATCH -D ./
#SBATCH -J jax_multi
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=110000
#SBATCH --constraint="apu"
#SBATCH --gres=gpu:1
#SBATCH --account=mpa_apu
#SBATCH --time=01:00:00
#SBATCH --mail-type=BEGIN,FAIL
#SBATCH --mail-user=uli@mpa-garching.mpg.de

module purge
module load gcc/14 openmpi/5.0 rocm/7.0

source ~/venv_jax_rocm7.0/bin/activate
export JAX_PLATFORM_NAME=rocm
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# JAX distributed coordinator setup
NODES=($(scontrol show hostname $SLURM_JOB_NODELIST))
HEAD_NODE=${NODES[0]}
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)

JAX_COORDINATOR_PORT=12345
export JAX_COORDINATOR_ADDRESS="${HEAD_NODE_IP}:${JAX_COORDINATOR_PORT}"

echo "==================================================================="
echo "JAX Multi-Node MI300A (ROCm)"
echo "==================================================================="
echo "Head node: $HEAD_NODE ($HEAD_NODE_IP)"
echo "All nodes: ${NODES[@]}"
echo "Total GPUs: ${#NODES[@]}"
echo "JAX_COORDINATOR_ADDRESS: $JAX_COORDINATOR_ADDRESS"
echo "==================================================================="

srun python main.py
