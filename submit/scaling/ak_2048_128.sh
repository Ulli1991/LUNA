#!/bin/bash -l
#SBATCH -o slurm_logs/scal_ak_2048_128.out.%j
#SBATCH -e slurm_logs/scal_ak_2048_128.err.%j
#SBATCH -D ./
#SBATCH -J sA_2048_128
#SBATCH --partition=gpu1
#SBATCH --nodes=32
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=125000
#SBATCH --time=00:30:00
module purge; module load gcc/13 cuda/12.6 openmpi_gpu/5.0
OUT=/ptmp/uli/jax-hydro/scal_ak_2048_128; mkdir -p $OUT; cd $OUT
echo "=== AthenaK Sedov 2048^3, 128 ranks, RK2 ==="
srun --gpus-per-task=1 --gpu-bind=closest $EXE_OVERRIDE /raven/u/uli/athenak/build_gpu_sedov/src/athena \
  -i /raven/u/uli/athenak/inputs/hydro/sedov_luna_1024.athinput \
  time/integrator=rk2 time/nlim=300 \
  mesh/nx1=2048 mesh/nx2=2048 mesh/nx3=2048 \
  meshblock/nx1=256 meshblock/nx2=256 meshblock/nx3=256 \
  output2/dt=1e10
echo "=== done ==="
