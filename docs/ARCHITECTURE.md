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

1. **Configuration**: every knob is a field of the frozen `Config` dataclass
   in `luna/template_config.py` (copied to `config.py`, named on the command line,
   or built by a run script; `field=value` overrides follow either,
   `luna/setup/cli.py`).
   Grid size, test problem, physics switches, output cadence — it's all there;
   docs/CONFIG.md explains each section. A test problem's own numbers are not in
   Config: they are the `Params` of its IC module, reached as `cfg.problem`
   (`problem.amp=0.01` on the command line).
2. **Entry**: `python main.py [config.py] [field=value ...]` runs `main()` in
   `luna/main.py`: read the config (`luna/setup/cli.py configure`), start JAX
   (`luna/setup/jax_init.py initialize_jax_distributed`: SLURM ranks, float64),
   print the banner, then `run(cfg)`, which is four calls, in order:
   ```
   mesh    = build_mesh(cfg)             # luna/setup/mesh.py      how the data lies on devices
   init    = read_ics(cfg, mesh)         # luna/setup/ics.py       each rank reads its part
   physics = assemble(cfg, mesh, init)   # luna/setup/assemble.py  step, clock, gravity, sources
   state   = evolve(cfg, init, physics)  # luna/setup/evolve.py    the loop, outputs, the end
   ```
   Each step has its helpers in its own file: the dense device mesh in
   `mesh.py`, the gas grid (ICs or the restart's) in `ics.py`, the hydro step
   in `luna/setup/hydro.py` (built by `assemble`, its input behind an
   `optimization_barrier` so a larger jit cannot fuse the stages before it
   into it; docs/AUTODIFF.md), and everything the loop
   writes in `luna/setup/outputs.py`. Run scripts call `launch(cfg)`
   (`luna/setup/cli.py`), the same start-up followed by `run(cfg)`.
   The run is one of three KINDS (`cfg.run`: gas, dm, gas+dm; luna/setup/run_kind.py)
   with independent switches: comoving time (`CosmologicalIntegration`),
   refinement (`AMR`, its own chunked scan in `luna/setup/runner_amr.py`) and the
   adjoint tasks (`autodiff_mode`): `run(cfg, task)` hands the prepared run to
   `task(init, cfg, physics, is_main)`, which differentiates the rollout of
   `luna/autodiff/rollout.py`; the paper experiments are such tasks, one file
   each in `scripts/examples/autodiff/` (`cfg.autodiff_task` names one).
3. **The loop** (`luna/setup/evolve.py`): build the initial state, restore a
   restart, pick a stable `dt`, take one `step`, report, write, and at the end
   `Outputs.finish` (the final report and plots of a gas run). One step is a
   palindrome around the hydro update — every
   operator-split source and the gravity stage run their `before` half in
   order, then the compiled hydro step `U = step(U, dt)`, then their `after`
   halves in reverse — so every source is second order in time. The opt-in
   learned flux corrector (`cfg.sil_weights`, `Corrector` in
   `luna/sources/stages.py`) is the innermost stage: nothing before, the
   network right after the hydro, once with the full dt. Gravity
   (`luna/gravity/kdk.py`) is one Poisson solve per step for the gas and every
   particle species (`luna/species/`), kick-drift-kick, on either the whole grid of
   one device or a block mesh (`luna/gravity/backend.py`).
   **Restart** (`cfg.restart`): `read_ics` reads the dense gas grid, t and step
   from the grid checkpoint instead of building the ICs (straight onto the
   mesh, so the grid is never held twice), and `setup_radiation` reads the
   radiation field the same way. Everything else is `Outputs.restore`, one
   branch per checkpoint format: the grid checkpoint's extras (the OU forcing
   state), or the whole cosmological set (grid, particles, a, tau, step).
4. **Output**: HDF5 snapshot files in `cfg.output_dir` (readers in
   `luna/file_io/`), figures via the scripts in `scripts/validation/plots/`.
5. **Result**: `main.run` returns a `RunResult` (final time, steps, output
   dir, extras).

## What lives where

| directory | contents |
|---|---|
| `luna/hydro/` | the core solver: reconstruction, Riemann solvers (flux of mass/momentum/energy between cells), the MUSCL-Hancock step |
| `luna/time_integration/` | assembles the per-step function from the hydro pieces |
| `luna/ic_generation/` | initial conditions for every test problem (registered by name) |
| `luna/setup/` | the run: mesh, initial conditions, `assemble`, `evolve`, outputs, the clock; plus the AMR scan |
| `luna/autodiff/` | the differentiable rollout (`rollout.py`: the production step in a remat'd `lax.scan`) and the linear-stability analysis built on it |
| `luna/sources/` | operator-split extra physics: forcing, cooling, external gravity, wind injection, orbits, the comoving floors |
| `luna/species/` | particle kinds in the gravity solve: dark matter, sink particles |
| `luna/gravity/` | Poisson solvers (multigrid), particle-mesh kernels, the KDK gravity stage and its two backends |
| `luna/cosmology/` | expansion background a(t), MUSIC initial-condition readers |
| `luna/amr/` | the block-structured adaptive-mesh layer (own design doc: `amr/DESIGN.md`) |
| `luna/utils/` | halo exchange (multi-GPU ghost cells), grid bookkeeping, plotting style |
| `luna/file_io/` | snapshot/checkpoint writers and readers |
| `tests/` | pytest suite (physics + parallelism regression gates) |
| `scripts/` | run scripts (`validation/runs/`), figure scripts (`validation/plots/`), and examples (`examples/autodiff/`: the adjoint experiments; `examples/runs/`: one-off runs no test or figure uses) |
| *(external)* | LUNA's `luna/amr/` is derived from **JAX-AMR** (Wen & Luo, MIT — https://github.com/JA4S/JAX-AMR, https://doi.org/10.48550/arXiv.2504.13750). Their source is NOT vendored here; licence in `src/luna/amr/LICENSE-JAX-AMR` and `amr/DESIGN.md` for what LUNA took vs. re-implemented. A reference checkout at `$LUNA_JAXAMR` (default `~/jaxamr`) lets one geometry test cross-validate against it; it skips when absent. |

## Suggested reading order

1. `luna/template_config.py` — skim the fields; this is the vocabulary of a run.
2. `luna/main.py` — the four calls above.
3. `luna/setup/evolve.py` — the loop and the step (about 120 lines). Then
   `luna/setup/assemble.py` for what the step is built from (the hydro in
   `luna/setup/hydro.py`), `luna/setup/outputs.py` for what the loop writes and
   restores, and `luna/gravity/kdk.py` for how gravity and the particles sit
   inside it.
4. `luna/hydro/muscl_hancock.py` — the numerical heart: how cell values are
   reconstructed, fluxes computed, and the state updated. Read alongside the
   glossary below.
5. `luna/utils/halo_exchange_shardmap.py` — how multi-GPU works: each GPU owns a
   tile of the grid and trades one-cell-thick boundary slabs with neighbours.
6. Then, per interest: `luna/sources/` (extra physics), `luna/gravity/poisson_multigrid.py`
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
  accurate in space and time for the pure hydro step; the operator-split
  sources are applied half before and half after it (Strang splitting), which
  keeps the whole step second order in time.
- **CFL condition / dt**: the time step must be small enough that no signal
  crosses more than a fraction of a cell per step, or the scheme blows up. LUNA
  sums the signal speeds over the directions, as RAMSES does (`CFL` 0.8;
  [HYDRO_TIMESTEP.md](HYDRO_TIMESTEP.md)).
- **Ghost / halo cells**: extra cells padded around a tile holding the
  neighbouring tile's (or the boundary condition's) values, so the stencil can
  be evaluated at the edge. On multiple GPUs the halo is exchanged every step.
- **Operator splitting**: apply each extra piece of physics (gravity kick,
  cooling, forcing) for half a step, advance the hydro, then apply them again
  in reverse order for the other half.
- **shard_map / ppermute**: the JAX primitives LUNA uses for multi-GPU — each
  device runs the step on its own tile (`shard_map`) and halos travel by a ring
  exchange (`ppermute`). Because these are traced JAX ops, gradients flow
  through them.
- **JIT / compile**: JAX traces the Python step function once into one compiled
  GPU program; the first step is slow (compilation), the rest are fast. Changing
  array shapes triggers recompilation.

## Two kernel paths: XLA for gradients, Pallas for production

Every hot kernel (hydro step, gravity kick, M1 sweep and ionization) exists twice.

- **XLA** (the default, `hydro_kernel = gravity_kernel = rad_kernel = "xla"`) is plain
  `jax.numpy`: portable (CPU, any GPU), differentiable, and the reference every test
  compares against. It is kept reasonably quick, never at the cost of autodiff.
- **Pallas** (`"pallas"`, or `"auto"` = Pallas on a GPU) is the hand-tuned Triton fast path
  for production runs: forward only, no VJP (autodiff runs always take XLA). On one A100
  (2026-10): hydro 1.2-2x XLA, M1 radiation 3.5x, and the only path that fits 512^3 FP64
  per GPU (`docs/history/CODE_NOTES.md`, A100 tuning).

The two must agree, bitwise or to round-off, and tests enforce it
(`tests/test_hydro_pallas.py`, `tests/test_radiation_m1_pallas.py`, `tests/test_kick_pallas.py`): a Pallas production
run is exactly the model you can differentiate with XLA. Speed work goes into Pallas;
changes to the XLA path keep the autodiff tests green. Which hydro branch runs for which
switches: `docs/HYDRO_KERNEL_PATHS.md`.

## Gravity: one solve per step

`luna/gravity/backend.py` has two backends with the same calls:
`LocalGravity` (the whole grid on one device) and `ShardedGravity` (a
(px, py, pz) block mesh, every kernel a shard_map program with explicit
collectives). The calls: `source(U, rho_dm, p, mass, active) -> (src, stats)`;
`solve(src, seed, prev, q) -> psi` (lap psi = src, unit coefficient; warm
from `seed + q (seed - prev)`, see docs/GRAVITY.md); `grad`,
`deposit`, `gather`; `kick_gas`, `kick_drift`, `close_kick`; and
`exchange`/`settle`/`resort` for particle ownership (no-ops on one device).

The source is the overdensity delta (comoving, kick scale
(3/2) Omega_m H0^2 a) or 4 pi G (rho - <rho>) (proper time, kick scale 1).
psi is solved with a unit coefficient and scaled at kick time by
`clock.kick_scale(a)`, so the carried psi stays a good warm seed across the
expansion. The KDK stage (`kdk.py`): `before` half-kicks the gas on the
carried potential, then each species half-kicks and drifts; `after` does one
solve at the new time, half-kicks the gas, lets each species gather its force
and close its kick, and last runs the events that are not dt-scaled (sink
accretion, creation).

## Particle species

Every species in `luna/species/` answers the same calls, so
`gravity.kdk.Gravity` never asks which one it holds: `assemble` (host arrays
-> device buffers), `deposit` (its mass on the grid, or None), `field_on_gas`
(an analytic field added to the gas kick, as a program `(fn, args)`, or
None), `before` (half-kick + drift), `force`, `after` (closing half-kick),
`finish` (what is not dt-scaled: accretion, creation), `dt_limit`, `check`
and `snapshot`.

Dark matter (`species/dm.py`): CIC deposit into the joint Poisson solve,
4th-order force gathered back, KDK with the force carried across the step.
On a device mesh each block owns its particles; they are exchanged when their
cumulative drift since the last exchange could leave the one-cell ghost
margin (0.45 dx, HACC-style), and every few exchanges a resort keeps the
buffers cell-ordered. IDs survive both, so snapshots match particle by
particle. Every exchange checks overflow, count and mass and raises if any
changed. Sinks: docs/TORCH.md.

## Comoving runs: super-comoving variables

Cosmological runs use super-comoving variables (RAMSES; Martel & Shapiro
1998): rho~ = a^3 rho, v~ = a v_pec, P~ = a^5 P, with time dtau = dt/a^2. In
them the comoving Euler equations are the plain Euler equations, so the flux
kernel is unchanged. The only explicit a-dependence is the Poisson source,
lap(phi~) = (3/2) Omega_m a delta (`Cosmology.poisson_coeff`); a(tau) comes
from a table (`build_a_of_tau`). MUSIC gas is meshed with v~ = a^{3/2} u
(Gadget u) and e~ = a^2 u_int.

## assemble, the loop and the step (setup/)

`assemble(cfg, mesh, kind, layout, init) -> Physics` builds the run's physics in one place:

- hydro: `step(U, dt)` and `compute_dt(U)` (setup/hydro.py); None for dark matter only
- clock: proper time or super-comoving tau (setup/clock.py)
- gravity: gas self-gravity and every particle species around one Poisson solve per step
  (gravity/kdk.py); None when off
- stages: the operator-split terms in order; the step runs their `before` in this order and
  their `after` reversed
- projections: what is not dt-scaled and runs once at the end of a step (the EOS
  projections, the floor stack, proper or comoving)

Four switches decide what is built: the run kind (gas / dm / gas+dm), the layout (one device
holds the whole grid, or a block mesh), comoving or proper time, and which sources and species
the config turns on.

`evolve(cfg, init, physics, mesh) -> State` is the one time loop. Each iteration picks dt (the
hydro's CFL, every stage's own limit such as the particles' acceleration criterion or the
wind's injection, the clock's caps, then clamped to land exactly on the next snapshot or the
end), takes one `step`, checks, reports and writes. `step` is a pure function of the state, so
the adjoint runner can scan it; `Outputs` (setup/outputs.py) restores a restart before the
first step and ends the run after the last. One step is a palindrome around the hydro update:

    for stage in stages:            stage.before(state, h, dt)
    U = hydro.step(U, dt)
    for stage in reversed(stages):  stage.after(state, h, dt)
    for p in projections:           U = p(U, a)

so every operator-split term is second order in time, and what is not dt-scaled (EOS
projections, floors, sink accretion and creation) runs once. The opening gravity solve comes
after a possible restore. `release_initial` drops the initial gas grid and the stages'
initial carries once the state owns them: `init` lives in the caller's frame for the whole run
and each stage keeps its carry0, so they held a U and a radiation carry, 11 GiB per GPU at
512^3, on the device for nothing (1024^3 run, 8 GPUs: 15 GiB in use before step 1, 26 after).

setup/hydro.py: inlined into a larger jit (autodiff scans, SiL training), the hydro step's
input sits behind an optimization barrier so the stages before it are materialised first;
fused, the OU kick came out wrong in all of U[4] at 1024^3 on 2x2x2 (docs/AUTODIFF.md, "What
the reference found"). Called on its own it is the same program as the bare step. The warm-up
compiles compute_dt and step without advancing U (an advancing warm-up left a phase offset).

## Other documents

- `README.md` — install, quick start, feature overview
- `docs/GALLERY.md` — results and validation figures with captions
- `docs/ENV.md` — environment-variable knobs (communication/perf modes)
- `amr/DESIGN.md` — the AMR layer's design and status
- `docs/history/` — dated plans, surveys and audits (the reasoning behind
  past decisions; not a description of the code today)
