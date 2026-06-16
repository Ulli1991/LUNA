#!/bin/bash -l
#SBATCH -o slurm_logs/scal_ak_fp64_64_2.out.%j
#SBATCH -e slurm_logs/scal_ak_fp64_64_2.err.%j
#SBATCH -D ./
#SBATCH -J Af64_64_2
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=60000
#SBATCH --time=00:20:00
module purge; module load gcc/13 cuda/12.6 openmpi_gpu/5.0
OUT=/ptmp/uli/jax-hydro/scal_ak_fp64_64_2; mkdir -p $OUT; cd $OUT
echo "=== AthenaK FP64 Sedov 64^3, 2 ranks, RK2 ==="
mb=(32 64 64)
srun --gpus-per-task=1 --gpu-bind=closest $EXE_OVERRIDE /raven/u/uli/athenak/build_gpu_sedov/src/athena \
  -i /raven/u/uli/athenak/inputs/hydro/sedov_luna_1024.athinput \
  time/integrator=rk2 time/nlim=300 \
  mesh/nx1=64 mesh/nx2=64 mesh/nx3=64 \
  meshblock/nx1=${mb[0]} meshblock/nx2=${mb[1]} meshblock/nx3=${mb[2]} \
  output2/dt=1e10
echo "=== done ==="
