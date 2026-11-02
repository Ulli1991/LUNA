# Changelog

All notable changes to LUNA. Versions follow [Semantic Versioning](https://semver.org); while
the version is 0.x, minor releases may change the configuration and the Python API.

## 0.1.0 (2026-11-02)

The first public release. Status words as in the README: validated (asserted by a test
against an independent reference), shown (runs and figures, no asserting test), experimental
(under development). Details and tests: docs/CLAIMS.md.

### Hydrodynamics
- Euler equations in 1D, 2D and 3D: unsplit MUSCL-Hancock, PLM reconstruction (minmod, MC,
  van Leer), HLLC and HLL Riemann solvers; FP32 by default, FP64 with `enable_x64=True`. Validated.
- Non-physical Hancock face states are refused with the face named; a first-order face
  fallback exists only as an opt-in (`allow_first_order_faces=True`) and is counted and reported.
- XLA kernel (default, differentiable reference) and an opt-in Pallas GPU kernel
  (`hydro_kernel="pallas"`), bitwise equal to the XLA kernel on an A100.
- Known open bug: HLLC carbuncle at grid-aligned strong shocks (strict xfail tests); HLL is
  the user's choice there, never switched automatically.

### Parallel runs
- Domain decomposition over many devices and processes (`MESH_SHAPE`); a sharded run
  reproduces the single-device run (bitwise in the Sedov and Kelvin-Helmholtz tests). Validated.
- Weak scaling measured to 4096 NVIDIA GH200 GPUs on JUPITER (FP32); runs on A100, GH200 and
  MI300A. Shown (measurements, not tests).

### Physics modules
- Self-gravity with periodic and isolated boundaries, energy-conserving coupling. Validated.
- Comoving cosmological integration (adiabatic expansion, top-hat collapse validated); dark
  matter particles with particle-mesh gravity (power spectrum compared with Arepo: shown).
- Radiative cooling from tables (tables validated against published equilibria; cooling in the
  step shown).
- Chemistry networks (chemcool networks 5 and 17 ported to JAX), radiation transport (M1 and
  M2 closures), sink particles and star formation coupled to Torch/AMUSE, block AMR, Helmholtz
  EOS and aprox13 nuclear network (standalone). Experimental.
- Starburst-driven wind (M82 disk, CC85 injection) and driven turbulence setups. Shown.

### Differentiation
- Reverse-mode gradients through the solver, checked against finite differences; known gaps
  listed in docs/AUTODIFF.md. Shown.

### Running and packaging
- One command line for every run: `luna run_file.py field=value ...` (or `python main.py` in a
  clone); every setting is a field of one `Config` (docs/CONFIG.md); unknown fields are errors.
- `pip install` extras `cuda12`, `rocm`, `plot`, `test`, `notebooks`; `luna.__version__`; run
  provenance (settings, command line, code revision or installed version) in `run.json`.
- Snapshots as HDF5 directories, checkpoints and restarts, `history.csv`.

### Tests
- Fast tier on every push, the full CPU suite (12 shards, 8 virtual devices), the 2-device
  files and the Torch/AMUSE coupling tests (AMUSE v2026.3.0 built from source) in public CI; GPU and multi-node tiers on the
  author's clusters. Reference solutions (AthenaK, Arepo, RAMSES, Pencil Code digitisations,
  AMReX Microphysics, spectral runs, LUNA's own regression outputs) in the separate,
  checksummed luna-reference-solutions repository; a missing or altered file fails, never skips.

### Known issues in 0.1.0
- Some full-suite CI tests fail at this release; fixes are under review for 0.1.1: AMR
  sharded vs single-device ghost cells, the MC-limiter accretion shocks in the coupled cosmology
  test, round-off differences between CPUs in the sink field and the stellar golden, the
  isolated-gravity multipole sum order, and outdated test stubs.
- The chemistry tests need the chemcool references, which move into luna-reference-solutions
  for 0.1.1; until then they run only where `$LUNA_DATA` holds them.

