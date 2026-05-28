#!/bin/bash -l
#SBATCH -o ./athenak_gpu_build.out.%j
#SBATCH -e ./athenak_gpu_build.err.%j
#SBATCH -D ./
#SBATCH -J ak_gpubuild
#SBATCH --partition=gpudev
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=60000
#SBATCH --time=00:15:00
#SBATCH --mail-type=BEGIN,FAIL
#SBATCH --mail-user=uli@mpa-garching.mpg.de

# Rebuild AthenaK with CUDA backend (Ampere80) using the kh_luna pgen.
# (Single GPU build node — Kokkos's nvcc_wrapper compiles offline.)

module purge
module load gcc/13 cuda/12.6 openmpi_gpu/5.0    # CUDA-aware OpenMPI

set -e
rm -rf /raven/u/uli/athenak/build_gpu
mkdir -p /raven/u/uli/athenak/build_gpu
cd /raven/u/uli/athenak/build_gpu

CC=mpicc CXX=/raven/u/uli/athenak/kokkos/bin/nvcc_wrapper cmake \
   -DCMAKE_BUILD_TYPE=Release \
   -DKokkos_ENABLE_CUDA=ON \
   -DKokkos_ARCH_AMPERE80=ON \
   -DAthena_ENABLE_MPI=ON \
   -DPROBLEM=kh_luna \
   ..

nice make -j8

echo "=== AthenaK GPU build done ==="
ls -lh src/athena
