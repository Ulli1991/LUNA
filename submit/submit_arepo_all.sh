#!/bin/bash -l
#SBATCH -o slurm_logs/arepo_all.out.%j
#SBATCH -e slurm_logs/arepo_all.err.%j
#SBATCH -D ./
#SBATCH -J arepo_all
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36
#SBATCH --cpus-per-task=1
#SBATCH --mem=120000
#SBATCH --time=04:00:00

# Runs all 4 AREPO gravity tests sequentially on one node with 36 MPI ranks.

AREPO_DIR=/raven/u/uli/arepo

module purge
module load intel/19.1.2 impi/2019.8 fftw-mpi/2.1.5 gsl/2.4 hdf5-mpi/1.8.21 hwloc/2.12
source ~/jax-env/bin/activate

run_test() {
    local TEST="$1"
    local BIN="$2"
    local EXAMPLE="$AREPO_DIR/examples/$TEST"
    local RUNDIR="$AREPO_DIR/run/examples/$TEST"
    echo "================ $TEST ================"
    mkdir -p "$RUNDIR/output"
    cd "$RUNDIR"
    [ -f IC.hdf5 ] || python3 "$EXAMPLE/create.py" "$RUNDIR"
    cp "$EXAMPLE/param.txt" .
    srun "$AREPO_DIR/$BIN" param.txt
    echo "================ $TEST done ================"
}

run_test evrard_3d Arepo_evrard
run_test bonnor_ebert_3d_sub Arepo_iso
run_test bonnor_ebert_3d_super Arepo_iso
# run_test bb79_3d Arepo_iso   # dropped: Voronoi insert_point stack-full at collapse (voronoi_3d.c:1802)
