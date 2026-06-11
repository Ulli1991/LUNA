#!/bin/bash -l
#SBATCH -o slurm_logs/arepo_evrard_hires.out.%j
#SBATCH -e slurm_logs/arepo_evrard_hires.err.%j
#SBATCH -D ./
#SBATCH -J arepo_evrard_hires
#SBATCH --partition=general
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=72
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00

# Evrard 256^3 ~ 16.7M cells, near Springel 2010 Fig 41 highest res.

AREPO_DIR=/raven/u/uli/arepo
TEST=evrard_3d
EXAMPLE="$AREPO_DIR/examples/$TEST"
RUNDIR="$AREPO_DIR/run/examples/${TEST}_hires"

module purge
module load intel/19.1.2 impi/2019.8 fftw-mpi/2.1.5 gsl/2.4 hdf5-mpi/1.8.21 hwloc/2.12
export LD_LIBRARY_PATH="$HDF5_HOME/lib:$GSL_HOME/lib:$FFTW_HOME/lib:$HWLOC_HOME/lib:$LD_LIBRARY_PATH"

mkdir -p "$RUNDIR/output"
cd "$RUNDIR"
cp "$EXAMPLE/param.txt" .
# move 256^3 IC into hires run dir if not already
[ -f IC.hdf5 ] || cp "$AREPO_DIR/run/examples/$TEST/IC.hdf5" .

srun "$AREPO_DIR/Arepo_evrard" param.txt
