#!/bin/bash
# Run the AthenaK soundwave Nx-scan on the login node (CPU build).
#
# For each Nx ∈ {128…4096} run TWO AthenaK simulations:
#   * PLM  — 2nd-order MUSCL, matches LUNA's order (fair 1-to-1 comparison)
#   * PPMX — 4th-order PPM with Colella–Sekora extrema-preserving limiters
#            (high-order showcase, shows what we'd gain by going beyond PLM)
#
# Final-time .bin files are copied to:
#   reference/athenak/soundwave_Nx{Nx}_plm.bin
#   reference/athenak/soundwave_Nx{Nx}_ppmx.bin

set -euo pipefail

module purge
module load gcc/14

ATHENAK_BIN=/raven/u/uli/athenak/build_cpu/src/athena
RUN_ROOT=/raven/u/uli/athenak/runs
REF_DIR="$(pwd)/reference/athenak"

mkdir -p "$REF_DIR"

write_input() {
    local NX=$1 SCHEME=$2 NGHOST=$3 OUTFILE=$4
    cat > "$OUTFILE" <<EOF
# AthenaK 1D acoustic eigenmode (right-going), Nx=${NX}, reconstruct=${SCHEME}.

<job>
basename = soundwave_Nx${NX}_${SCHEME}

<mesh>
nghost = ${NGHOST}
nx1    = ${NX}
x1min  = 0.0
x1max  = 1.0
ix1_bc = periodic
ox1_bc = periodic

nx2    = 1
x2min  = -0.5
x2max  = 0.5
ix2_bc = periodic
ox2_bc = periodic

nx3    = 1
x3min  = -0.5
x3max  = 0.5
ix3_bc = periodic
ox3_bc = periodic

<meshblock>
nx1 = ${NX}
nx2 = 1
nx3 = 1

<time>
evolution  = dynamic
integrator = rk2
cfl_number = 0.4
nlim       = -1
tlim       = 1.0
ndiag      = 10000

<hydro>
eos         = ideal
reconstruct = ${SCHEME}
rsolver     = hllc
gamma       = 1.66666666666667

<problem>
pgen_name = linear_wave
wave_flag = 4
amp       = 1.0e-4
dens      = 1.0
pgas      = 0.6
vx0       = 0.0
along_x1  = true

<output1>
file_type = bin
variable  = hydro_w
dt        = 1.0
ghost_zones = false
EOF
}

for NX in 128 256 512 1024 2048 4096; do
    # ---- PLM (2 ghost zones) ----
    RUNDIR="${RUN_ROOT}/soundwave_Nx${NX}_plm"
    mkdir -p "$RUNDIR"
    write_input "$NX" plm 2 "${RUNDIR}/soundwave.athinput"
    echo "=== Nx=${NX} PLM ==="
    ( cd "$RUNDIR" && "$ATHENAK_BIN" -i soundwave.athinput 2>&1 | tail -3 )
    cp "${RUNDIR}/bin/soundwave_Nx${NX}_plm.hydro_w.00001.bin" \
       "${REF_DIR}/soundwave_Nx${NX}_plm.bin"

    # ---- PPMX (3 ghost zones) ----
    RUNDIR="${RUN_ROOT}/soundwave_Nx${NX}_ppmx"
    mkdir -p "$RUNDIR"
    write_input "$NX" ppmx 3 "${RUNDIR}/soundwave.athinput"
    echo "=== Nx=${NX} PPMX ==="
    ( cd "$RUNDIR" && "$ATHENAK_BIN" -i soundwave.athinput 2>&1 | tail -3 )
    cp "${RUNDIR}/bin/soundwave_Nx${NX}_ppmx.hydro_w.00001.bin" \
       "${REF_DIR}/soundwave_Nx${NX}_ppmx.bin"
done

echo
echo "AthenaK reference files in ${REF_DIR}:"
ls -la "${REF_DIR}"
