#!/bin/bash -l
#SBATCH -o slurm_logs/arepo_bb79_iso.out.%j
#SBATCH -e slurm_logs/arepo_bb79_iso.err.%j
#SBATCH -D ./
#SBATCH -J arepo_bb79_iso
#SBATCH --partition=general
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=72
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00

AREPO_DIR=/raven/u/uli/arepo
EXAMPLE=$AREPO_DIR/examples/bb79_3d
RUNDIR=$AREPO_DIR/run/examples/bb79_3d_iso

module purge
module load intel/19.1.2 impi/2019.8 fftw-mpi/2.1.5 gsl/2.4 hdf5-mpi/1.8.21 hwloc/2.12
export LD_LIBRARY_PATH="$HDF5_HOME/lib:$GSL_HOME/lib:$FFTW_HOME/lib:$HWLOC_HOME/lib:$LD_LIBRARY_PATH"

cd "$RUNDIR"
cp "$EXAMPLE/param.txt" .

srun "$AREPO_DIR/Arepo_iso_iso" param.txt
