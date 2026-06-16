#!/bin/bash -l
#SBATCH -o slurm_logs/scal_ak_fp64_256_8.out.%j
#SBATCH -e slurm_logs/scal_ak_fp64_256_8.err.%j
#SBATCH -D ./
#SBATCH -J Af64_256_8
#SBATCH --partition=gpu1
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=125000
#SBATCH --time=00:20:00
module purge; module load gcc/13 cuda/12.6 openmpi_gpu/5.0
OUT=/ptmp/uli/jax-hydro/scal_ak_fp64_256_8; mkdir -p $OUT; cd $OUT
echo "=== AthenaK FP64 Sedov 256^3, 8 ranks, RK2 ==="
mb=(128 128 128)
srun --gpus-per-task=1 --gpu-bind=closest $EXE_OVERRIDE /raven/u/uli/athenak/build_gpu_sedov/src/athena \
  -i /raven/u/uli/athenak/inputs/hydro/sedov_luna_1024.athinput \
  time/integrator=rk2 time/nlim=300 \
  mesh/nx1=256 mesh/nx2=256 mesh/nx3=256 \
  meshblock/nx1=${mb[0]} meshblock/nx2=${mb[1]} meshblock/nx3=${mb[2]} \
  output2/dt=1e10
echo "=== done ==="
