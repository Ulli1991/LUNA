#!/bin/bash -l
#SBATCH -o ./athenak_sedov_build.out.%j
#SBATCH -e ./athenak_sedov_build.err.%j
#SBATCH -D ./
#SBATCH -J ak_sedovbuild
#SBATCH --partition=gpudev
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=60000
#SBATCH --time=00:15:00
#SBATCH --mail-type=BEGIN,FAIL
#SBATCH --mail-user=uli@mpa-garching.mpg.de

# Separate AthenaK build with the sedov_luna pgen. Leaves the existing
# build_gpu/ (kh_luna) intact for the queued KH reference run.

module purge
module load gcc/13 cuda/12.6 openmpi_gpu/5.0

set -e
rm -rf /raven/u/uli/athenak/build_gpu_sedov
mkdir -p /raven/u/uli/athenak/build_gpu_sedov
cd /raven/u/uli/athenak/build_gpu_sedov

CC=mpicc CXX=/raven/u/uli/athenak/kokkos/bin/nvcc_wrapper cmake \
   -DCMAKE_BUILD_TYPE=Release \
   -DKokkos_ENABLE_CUDA=ON \
   -DKokkos_ARCH_AMPERE80=ON \
   -DAthena_ENABLE_MPI=ON \
   -DPROBLEM=sedov_luna \
   ..

nice make -j8

echo "=== AthenaK Sedov build done ==="
ls -lh src/athena
