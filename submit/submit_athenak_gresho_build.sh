#!/bin/bash -l
#SBATCH -o slurm_logs/athenak_gresho_build.out.%j
#SBATCH -e slurm_logs/athenak_gresho_build.err.%j
#SBATCH -D ./
#SBATCH -J ak_gresho_build
#SBATCH --partition=gpudev
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=60000
#SBATCH --time=00:15:00

# Build AthenaK with gresho_luna pgen.

module purge
module load gcc/13 cuda/12.6 openmpi_gpu/5.0

set -e
rm -rf /raven/u/uli/athenak/build_gpu_gresho
mkdir -p /raven/u/uli/athenak/build_gpu_gresho
cd /raven/u/uli/athenak/build_gpu_gresho

CC=mpicc CXX=/raven/u/uli/athenak/kokkos/bin/nvcc_wrapper cmake \
   -DCMAKE_BUILD_TYPE=Release \
   -DKokkos_ENABLE_CUDA=ON \
   -DKokkos_ARCH_AMPERE80=ON \
   -DAthena_ENABLE_MPI=ON \
   -DPROBLEM=gresho_luna \
   ..

nice make -j8

echo "=== AthenaK gresho build done ==="
ls -lh src/athena
