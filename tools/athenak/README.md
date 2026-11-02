# AthenaK from LUNA's initial data

`pgen/luna_ic.cpp` is the AthenaK problem generator that reads a LUNA IC directory (`U.npy` (5, Nx, Ny, Nz)
float64 + `meta.json`, as `src/luna/ic_generation/ic_file.py` and `tools/luna_ic.py` write it) into the
conserved variables, each rank only its own MeshBlocks, uniform meshes, 1D/2D/3D. It writes float64 dumps
in the same layout (every history output with `dump = true`, and at the end), and with `errors = true`
AthenaK's own `OutputErrors` against the IC. Options are in the file header.

Copy of the AthenaK branch `luna_ic_a6` (branch `luna_ic` of an AthenaK checkout `$ATHENAK`, base 886dd2a, plus the static
potential below; without `potential` it behaves as `luna_ic`). Build (A100, Raven), on a
GPU node, ~3 min with 18 cores:

```bash
module purge; module load gcc/13 cuda/12.6 openmpi_gpu/5.0
cp tools/athenak/pgen/luna_ic.cpp $ATHENAK/src/pgen/
cmake -S $ATHENAK -B $ATHENAK/build_gpu_lunaic -DKokkos_ENABLE_CUDA=On -DKokkos_ARCH_AMPERE80=On \
  -DAthena_ENABLE_MPI=On -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=$ATHENAK/kokkos/bin/nvcc_wrapper -DPROBLEM=luna_ic
cmake --build $ATHENAK/build_gpu_lunaic -j 18
```

Runs and reductions: `tools/athenak_refs.py` (prep writes ICs and inputs from the tests' own `initial()`,
reduce applies the tests' metric functions); the references it made are in `luna-reference-solutions/athenak/`.
1D MeshBlocks above 512 cells abort in AthenaK ("insufficient shared memory"): split them.
Strong shocks with PPMX/PPM4/WENOZ need `dfloor`/`pfloor` in `<hydro>` (`prep ... floor=1e-8`): the defaults are
FLT_MIN and give NaN at the Woodward-Colella collision, while AthenaK still ends "normally" at tlim.
The binary used so far (sha256 e40b2abd..., 2026-10-11 15:24) predates the luna_ic commits c0d1427d/c52f88f0
(pgen only); `luna-reference-solutions/athenak/README.md` has the details.

Self-gravity (A7, A8) needs upstream AthenaK: 886dd2a has none; the multigrid solver came with PR #776
(8a6a8efa). Branch `luna_ic_mg` = upstream main c5a0d7f9 + the three luna_ic commits cherry-picked (7f1c7989),
in its own git worktree `$ATHENAK_MG` (so the `luna_ic` checkout and `build_gpu_lunaic` stay untouched),
Kokkos submodule 6739bc62 (4.7). Built 2026-10-14 on a gpu1 node in 4 min, 18 cores (job 31167052):

```bash
module purge; module load gcc/13 cuda/12.6 openmpi_gpu/5.0
A=$ATHENAK_MG                  # git -C $ATHENAK worktree add -b luna_ic_mg $A origin/main; cherry-pick; submodule update
cmake -S $A -B $ATHENAK/build_gpu_lunaic_mg -DKokkos_ENABLE_CUDA=On -DKokkos_ARCH_AMPERE80=On \
  -DAthena_ENABLE_MPI=On -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=$A/kokkos/bin/nvcc_wrapper -DPROBLEM=luna_ic
cmake --build $ATHENAK/build_gpu_lunaic_mg -j 18
```

Binary `build_gpu_lunaic_mg/src/athena`, sha256 `fd151e3a870b8d0a8c697c966ddef1aec10e25b4fdd924d429ff51b46dcf97f2`.
Regression (job 31167109): A1 PLM + RK2 N = 128 bitwise equal to the old binary's dump, N = 1024 to 2e-14
(RMS-L1 / EPS 8.184246e-5 vs 8.184263e-5); a non-PROBLEM build of the same tree (`build_gpu_mg_builtin`, sha256
866f06ae...) runs AthenaK's own `jeans_wave` at 64 x 32 x 32 with omega 16.430 (n_J 0.5) and 32.488 (n_J 2), as
the c5a0d7f check (job 31166749), and the same to 6 digits with `threshold = 0`. Run with
`ATHENAK_DIR=$ATHENAK_MG ATHENAK_EXE=$ATHENAK/build_gpu_lunaic_mg/src/athena` so `athenak_refs.py`
records this tree and binary. `luna_ic.athinput(gravity=four_pi_G)` writes `<gravity>` (FMG, then V-cycles to
round-off: `threshold = 0`) and `self_gravity` in `<hydro_srcterms>`; MeshBlocks must be cubic powers of two.
In 3D, PLM + RK2 at CFL 0.4 is unstable (the A7 stable wave grows to O(0.1) by 128 cells, 125 steps): A7/A8 use 0.3.

Static potentials (A6, A17 later): AthenaK's `const_accel` is uniform only, so the pgen got a user source term,
`<problem> user_srcs = true, potential = sin, phi0, wavelength, axis`: phi = phi0 sin(2 pi x / wavelength),
g = -dphi/dx at the cell centres x = xmin + (i + 1/2) dx with i the global cell index (LUNA's
`external_potential = "sin"`, `src/luna/sources/external_gravity.py`), added in every RK stage as
`const_accel` does (bdt rho g to the momentum, bdt rho v g to the energy, from the stage primitives).
Branch `luna_ic_a6` = `luna_ic` c52f88f0 + 687f5a64, its own worktree `$ATHENAK_A6` (Kokkos 08ceff92, as
`luna_ic`), build `$ATHENAK/build_gpu_lunaic_a6` (same cmake line as `build_gpu_lunaic`, 2026-10-14, job
31171756, 4 min), binary sha256 `c6ded97ec8037041832297054f679c053457d1bb2fc36a60be883b143c3134cd`. The hydro
kernels are 886dd2a's. Run with `ATHENAK_DIR=$ATHENAK_A6 ATHENAK_EXE=$ATHENAK/build_gpu_lunaic_a6/src/athena`;
`luna_ic.athinput(potential=(phi0, wavelength, axis))` writes the block.
