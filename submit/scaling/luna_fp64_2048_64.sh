#!/bin/bash -l
#SBATCH -o slurm_logs/scal_luna_fp64_2048_64.out.%j
#SBATCH -e slurm_logs/scal_luna_fp64_2048_64.err.%j
#SBATCH -D ./
#SBATCH -J Lf64_2048_64
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --mem=200000
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
module purge; module load gcc/13 cuda/12
source ~/jax-env/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export JAX_PLATFORM_NAME=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_PREALLOCATE=false
NL=($(scontrol show hostname $SLURM_JOB_NODELIST)); HN=${NL[0]}
HNIP=$(srun --nodes=1 --ntasks=1 -w "$HN" hostname --ip-address)
export JAX_COORDINATOR_ADDRESS="${HNIP}:12345"
export LUNA_X64=1
export LUNA_SEDOV_NX=2048
export LUNA_USE_SHARDMAP=1
export LUNA_MESH_SHAPE=4,4,4
export LUNA_SEDOV_NSNAPS=1
export LUNA_SEDOV_OUT=/ptmp/uli/jax-hydro/snapshots/scal_luna_fp64_2048_64
mkdir -p "$LUNA_SEDOV_OUT"
srun python -u -m scripts.validation.runs.sedov3d_run
