# How LUNA fits together — a guided tour

This page is for someone new to the code (some Python, not necessarily any
computational-fluid-dynamics background). It tells you what happens when LUNA
runs, what lives where, and in what order to read the code.

## What LUNA is

LUNA solves the equations of gas dynamics (the Euler equations) on a 3D grid
of cells, on one or many GPUs, using [JAX](https://docs.jax.dev) — so the same
code is fast, parallel, and differentiable. Optional physics: self-gravity,
radiative cooling, dark-matter particles, cosmological expansion, and adaptive
mesh refinement (AMR).

## The 60-second tour of a run

1. **Configuration**: every knob is a field of the `Config` dataclass in
   `template_config.py` (copied to `config.py`, or overridden by a run script).
   Grid size, test problem, physics switches, output cadence — it's all there,
   with a comment per field.
2. **Entry**: `main.run(cfg)` (in `main.py`) looks at a handful of config
   flags and dispatches to ONE runner — read the `if` ladder in `main.run`,
   it is the map of the whole code:
   | flag | runner | what it evolves |
   |---|---|---|
   | *(none set)* | `setup/runner_distributed.py` | gas on a fixed grid (the standard path) |
   | `AMR` | `setup/runner_amr.py` | gas on an adaptively refined grid |
   | `CosmologicalIntegration` | `setup/runner_cosmo.py` | gas in an expanding universe |
   | `CosmoDM` | `setup/runner_cosmo_dm.py` | dark-matter particles only |
   | `CosmoHydro` | `setup/runner_cosmo_hydro.py` | gas + dark matter together |
   | `autodiff_mode` | `setup/runner_autodiff.py` | a differentiable run (gradients!) |
3. **The loop**: every runner is the same idea — a `while` loop that picks a
   stable time step `dt`, calls a compiled step function `U = step_fn(U, dt)`
   to advance the gas state one step, applies any extra physics
   (operator-split sources), and periodically writes a snapshot.
4. **Output**: HDF5 snapshot files in `cfg.output_dir` (readers in
   `file_io/`), figures via the scripts in `scripts/validation/plots/`.
5. **Result**: `main.run` returns a `RunResult` (final time, steps, output
   dir, extras).

## What lives where

| directory | contents |
|---|---|
| `hydro/` | the core solver: reconstruction, Riemann solvers (flux of mass/momentum/energy between cells), the MUSCL-Hancock step |
| `time_integration/` | assembles the per-step function from the hydro pieces |
| `ic_generation/` | initial conditions for every test problem (registered by name) |
| `setup/` | the runners (time loops) + distributed/mesh setup |
| `sources/` | operator-split extra physics: forcing, cooling, gravity kicks, sinks, particles |
| `gravity/` | Poisson solvers (multigrid) + particle-mesh gravity for dark matter |
| `cosmology/` | expansion background a(t), MUSIC initial-condition readers |
| `amr/` | the block-structured adaptive-mesh layer (own design doc: `amr/DESIGN.md`) |
| `utils/` | halo exchange (multi-GPU ghost cells), grid bookkeeping, plotting style |
| `file_io/` | snapshot/checkpoint writers and readers |
| `tests/` | pytest suite (physics + parallelism regression gates) |
| *(external)* | LUNA's `amr/` is derived from **JAX-AMR** (Wen & Luo, MIT — https://github.com/JA4S/JAX-AMR, https://doi.org/10.48550/arXiv.2504.13750). Their source is NOT vendored here; see `THIRD_PARTY_NOTICES.md` for the licence and `amr/DESIGN.md` for what LUNA took vs. re-implemented. A reference checkout at `$LUNA_JAXAMR` (default `~/jaxamr`) lets one geometry test cross-validate against it; it skips when absent. |

## Suggested reading order

1. `template_config.py` — skim the fields; this is the vocabulary of a run.
2. `main.py` — short; the dispatch ladder above.
3. `setup/runner_distributed.py` — the standard time loop: dt choice,
   step, sources, snapshots, checkpoints. Everything else is a variation.
4. `hydro/muscl_hancock.py` — the numerical heart: how cell values are
   reconstructed, fluxes computed, and the state updated. Read alongside the
   glossary below.
5. `utils/halo_exchange_shardmap.py` — how multi-GPU works: each GPU owns a
   tile of the grid and trades one-cell-thick boundary slabs with neighbours.
6. Then, per interest: `sources/` (extra physics), `gravity/poisson_multigrid.py`
   (self-gravity), `amr/DESIGN.md` (adaptive meshes), `docs/ENV.md` (tuning knobs).

## Ten pieces of jargon, quickly

- **U — conserved variables**: the state per cell, `[density, momentum (x,y,z),
  total energy]`. "Conserved" because fluxes move them between cells without
  creating or destroying totals.
- **W — primitive variables**: `[density, velocity, pressure]` — the friendly
  form used for reconstruction; converted back and forth from U.
- **Riemann problem / solver**: given the gas state on the left and right of a
  cell face, compute the flux through that face. LUNA's default approximate
  solver is HLLC.
- **Reconstruction / PLM / limiter**: estimate the value at a cell face from
  cell averages using a limited linear slope — "limited" so sharp jumps (shocks)
  don't produce spurious oscillations. Limiters: minmod (safest), MonCen, VanLeer.
- **MUSCL-Hancock**: the two-stage scheme LUNA uses — predict the state half a
  time step ahead, then compute fluxes with the predicted states. Second-order
  accurate in space and time.
- **CFL condition / dt**: the time step must be small enough that no signal
  crosses more than a fraction (`CFL` ≈ 0.4) of a cell per step, or the scheme
  blows up.
- **Ghost / halo cells**: extra cells padded around a tile holding the
  neighbouring tile's (or the boundary condition's) values, so the stencil can
  be evaluated at the edge. On multiple GPUs the halo is exchanged every step.
- **Operator splitting**: advance the hydro first, then apply each extra piece
  of physics (gravity kick, cooling, forcing) one after another within the step.
- **shard_map / ppermute**: the JAX primitives LUNA uses for multi-GPU — each
  device runs the step on its own tile (`shard_map`) and halos travel by a ring
  exchange (`ppermute`). Because these are traced JAX ops, gradients flow
  through them.
- **JIT / compile**: JAX traces the Python step function once into one compiled
  GPU program; the first step is slow (compilation), the rest are fast. Changing
  array shapes triggers recompilation.

## Other documents

- `README.md` — install, quick start, feature overview
- `docs/GALLERY.md` — results and validation figures with captions
- `docs/ENV.md` — environment-variable knobs (communication/perf modes)
- `amr/DESIGN.md` — the AMR layer's design and status
