#!/bin/bash -l
#SBATCH -o ./athenak_sedov.out.%j
#SBATCH -e ./athenak_sedov.err.%j
#SBATCH -D ./
#SBATCH -J ak_sedov
#SBATCH --partition=gpudev
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=125000
#SBATCH --time=00:15:00
#SBATCH --mail-type=BEGIN,FAIL
#SBATCH --mail-user=uli@mpa-garching.mpg.de

# AthenaK 3D Sedov 256^3 reference run, 4 MPI ranks / 4 A100 GPUs.

module purge
module load gcc/13 cuda/12.6 openmpi_gpu/5.0    # MPCDF CUDA-aware OpenMPI

OUTDIR=/raven/u/uli/jax-hydro/reference/athenak/sedov_256
mkdir -p ${OUTDIR}
cd ${OUTDIR}

echo "=== AthenaK Sedov 256^3, 4 ranks x 4 A100 ==="
nvidia-smi --query-gpu=name,memory.total --format=csv

srun --gpus-per-task=1 --gpu-bind=closest \
    /raven/u/uli/athenak/build_gpu_sedov/src/athena \
    -i /raven/u/uli/athenak/inputs/hydro/sedov_luna.athinput

echo "=== done ==="
ls -lh
