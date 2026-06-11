#!/bin/bash -l
#SBATCH -o slurm_logs/arepo_%x.out.%j
#SBATCH -e slurm_logs/arepo_%x.err.%j
#SBATCH -D ./
#SBATCH -J arepo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=60000
#SBATCH --time=01:30:00
#SBATCH --partition=interactive

# Usage: sbatch -J arepo_<test> --export=ALL,LUNA_AREPO_TEST=<test> submit_arepo.sh
#   where <test> is one of: evrard_3d, bonnor_ebert_3d_sub, bonnor_ebert_3d_super, bb79_3d

TEST="${LUNA_AREPO_TEST:?LUNA_AREPO_TEST not set}"
AREPO_DIR=/raven/u/uli/arepo
EXAMPLE="$AREPO_DIR/examples/$TEST"
RUNDIR="$AREPO_DIR/run/examples/$TEST"

module purge
module load intel/19.1.2 impi/2019.8 fftw-mpi/2.1.5 gsl/2.4 hdf5-mpi/1.8.21 hwloc/2.12
# Make sure runtime libs are found (modules set these, but be explicit).
export LD_LIBRARY_PATH="$HDF5_HOME/lib:$GSL_HOME/lib:$FFTW_HOME/lib:$HWLOC_HOME/lib:$LD_LIBRARY_PATH"

# Pick binary by test
case "$TEST" in
    evrard_3d)               BIN=Arepo_evrard ;;
    bonnor_ebert_3d_sub|bonnor_ebert_3d_super|bb79_3d)  BIN=Arepo_iso ;;
    *) echo "unknown test $TEST"; exit 1 ;;
esac

mkdir -p "$RUNDIR/output"
cd "$RUNDIR"

cp "$EXAMPLE/param.txt" .

# Run
srun "$AREPO_DIR/$BIN" param.txt
