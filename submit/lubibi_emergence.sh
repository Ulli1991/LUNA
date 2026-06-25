#!/bin/bash -l
#SBATCH -o slurm_logs/lubibi_emergence.out.%j
#SBATCH -e slurm_logs/lubibi_emergence.err.%j
#SBATCH -J lubibi_emerge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=60000
#SBATCH --time=01:00:00
module purge; module load gcc/13 cuda/12
source ~/jax-env/bin/activate
export JAX_PLATFORM_NAME=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
cd /raven/u/uli/jax-hydro
export LUNA_EMERG_NX=${LUNA_EMERG_NX:-256}
export LUNA_EMERG_NSTEPS=${LUNA_EMERG_NSTEPS:-300}
export LUNA_EMERG_NITER=${LUNA_EMERG_NITER:-200}
export LUNA_EMERG_IMAGE=/ptmp/uli/jax-hydro/emergence/lu_target.npy
echo "=== you+LUNA emerging from overlapping blast waves (2D, Nx=$LUNA_EMERG_NX) ==="
python -u -m scripts.validation.runs.blast_emergence_run
echo "=== done ==="
