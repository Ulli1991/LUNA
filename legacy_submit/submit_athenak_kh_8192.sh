#!/bin/bash -l
#SBATCH -o ./athenak_kh.out.%j
#SBATCH -e ./athenak_kh.err.%j
#SBATCH -D ./
#SBATCH -J ak_kh8192
#SBATCH --partition=gpu1
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=125000
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,FAIL
#SBATCH --mail-user=uli@mpa-garching.mpg.de

# AthenaK 8192^2 Kelvin-Helmholtz reference run, 4 MPI ranks / 4 A100 GPUs.
# Output: reference/athenak/kh_8192/

module purge
module load gcc/13 cuda/12.6 openmpi_gpu/5.0    # CUDA-aware OpenMPI

OUTDIR=/raven/u/uli/jax-hydro/reference/athenak/kh_8192
mkdir -p ${OUTDIR}
cd ${OUTDIR}

echo "=== AthenaK KH 8192^2, 4 ranks x 4 A100 ==="
nvidia-smi --query-gpu=name,memory.total --format=csv

srun --gpus-per-task=1 --gpu-bind=closest \
    /raven/u/uli/athenak/build_gpu/src/athena \
    -i /raven/u/uli/athenak/inputs/hydro/kh_luna_8192.athinput

echo "=== done ==="
ls -lh
