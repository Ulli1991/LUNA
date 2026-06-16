#!/bin/bash -l
#SBATCH -o slurm_logs/scal_ak_fp64_1024_64.out.%j
#SBATCH -e slurm_logs/scal_ak_fp64_1024_64.err.%j
#SBATCH -D ./
#SBATCH -J Af64_1024_64
#SBATCH --partition=gpu1
#SBATCH --nodes=16
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=125000
#SBATCH --time=00:25:00
module purge; module load gcc/13 cuda/12.6 openmpi_gpu/5.0
OUT=/ptmp/uli/jax-hydro/scal_ak_fp64_1024_64; mkdir -p $OUT; cd $OUT
echo "=== AthenaK FP64 Sedov 1024^3, 64 ranks, RK2 ==="
srun --gpus-per-task=1 --gpu-bind=closest $EXE_OVERRIDE /raven/u/uli/athenak/build_gpu_sedov/src/athena \
  -i /raven/u/uli/athenak/inputs/hydro/sedov_luna_1024.athinput \
  time/integrator=rk2 time/nlim=300 \
  mesh/nx1=1024 mesh/nx2=1024 mesh/nx3=1024 \
  meshblock/nx1=256 meshblock/nx2=256 meshblock/nx3=256 \
  output2/dt=1e10
echo "=== done ==="
