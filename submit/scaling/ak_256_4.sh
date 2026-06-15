#!/bin/bash -l
#SBATCH -o slurm_logs/scal_ak_256_4.out.%j
#SBATCH -e slurm_logs/scal_ak_256_4.err.%j
#SBATCH -D ./
#SBATCH -J sA_256_4
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=125000
#SBATCH --time=00:15:00
module purge; module load gcc/13 cuda/12.6 openmpi_gpu/5.0
OUT=/ptmp/uli/jax-hydro/scal_ak_256_4; mkdir -p $OUT; cd $OUT
echo "=== AthenaK Sedov 256^3, 4 ranks, RK2 ==="
srun --gpus-per-task=1 --gpu-bind=closest $EXE_OVERRIDE /raven/u/uli/athenak/build_gpu_sedov/src/athena \
  -i /raven/u/uli/athenak/inputs/hydro/sedov_luna_1024.athinput \
  time/integrator=rk2 time/nlim=300 \
  mesh/nx1=256 mesh/nx2=256 mesh/nx3=256 \
  meshblock/nx1=128 meshblock/nx2=128 meshblock/nx3=128 \
  output2/dt=1e10
echo "=== done ==="
