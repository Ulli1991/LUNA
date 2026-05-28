#!/bin/bash -l
#SBATCH -o ./athenak_sedov.out.%j
#SBATCH -e ./athenak_sedov.err.%j
#SBATCH -D ./
#SBATCH -J ak_sedov1024
#SBATCH --partition=gpu1
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=125000
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,FAIL
#SBATCH --mail-user=uli@mpa-garching.mpg.de

# AthenaK 3D Sedov 1024^3 reference: 2 nodes x 4 A100s = 8 MPI ranks.

module purge
module load gcc/13 cuda/12.6 openmpi_gpu/5.0    # MPCDF CUDA-aware OpenMPI

OUTDIR=/raven/u/uli/jax-hydro/reference/athenak/sedov_1024
mkdir -p ${OUTDIR}
cd ${OUTDIR}

echo "=== AthenaK Sedov 1024^3, 8 ranks x A100 ==="
nvidia-smi --query-gpu=name,memory.total --format=csv

srun --gpus-per-task=1 --gpu-bind=closest \
    /raven/u/uli/athenak/build_gpu_sedov/src/athena \
    -i /raven/u/uli/athenak/inputs/hydro/sedov_luna_1024.athinput

echo "=== done ==="
ls -lh
