#!/bin/bash -l
#SBATCH -o slurm_logs/mp_ic_test.out.%j
#SBATCH -e slurm_logs/mp_ic_test.err.%j
#SBATCH -J mp_ic_test
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=60000
#SBATCH --time=00:20:00
module purge; module load gcc/13
source ~/jax-env/bin/activate
NL=($(scontrol show hostname $SLURM_JOB_NODELIST)); HN=${NL[0]}
HNIP=$(srun --nodes=1 --ntasks=1 -w "$HN" hostname --ip-address)
export JAX_COORDINATOR_ADDRESS="${HNIP}:12346"
export JAX_PLATFORM_NAME=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
cd /raven/u/uli/jax-hydro
srun python -u scratch_mp_ic_test.py
