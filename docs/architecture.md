# LUNA architecture

One-page map of the codebase and the data flow through a time step.

## Directory layout

```
hydro/                Pure physics primitives. No JAX sharding, no I/O.
  fluxes.py           HLLC Riemann (1D/2D/3D)
  fluxes_lhllc.py     L-HLLC variant (Velasco-Romero & Teyssier 2025)
  slopes.py           PLM slope limiters (MonCen / minmod / VanLeer)
  jacobian.py         Primitive-variable Jacobian for the MUSCL-Hancock predictor
  muscl_hancock.py    Predictor + per-axis reconstruction + flux divergence
  conversions.py      cons<->prim conversions
  physics_checks.py   min/max/NaN diagnostics
  cooling.py          Sutherland-Dopita 93 CIE + Townsend 09 cooling
  external_gravity.py Miyamoto-Nagai disk + NFW halo accelerations
  turbulence.py       Ornstein-Uhlenbeck driver

ic_generation/        Initial-condition functions, one per problem.
  ic.py               Registry: test_problem string -> IC callable
  sod.py, kh.py, ...  Individual IC functions (slice-aware for sharding)

setup/                Distributed orchestration. JAX-aware, sharding-aware.
  runner_distributed.py    Main time loop + NaN guard + diagnostics + snapshots
  distributed_setup.py     JAX init, sharding mesh, IC sharding, integrator factory
  finalize_distributed.py  Final snapshot + post-run plots
  source_ou_forcing.py     Operator-split OU turbulence driver (per-step apply)
  source_cc85.py           Operator-split CC85 central injection (per-step apply)
  source_external_gravity.py  Operator-split MN+NFW gravity (per-step apply)
  source_cooling.py        Operator-split Townsend cooling (per-step apply)

time_integration/     Step factories. Produces (compute_dt_fn, step_fn).
  integrators.py      MUSCL-Hancock factory (only supported integrator)
  wb_integrator.py    Well-balanced MUSCL-Hancock for stratified flows

file_io/              Sharded HDF5 snapshot + checkpoint I/O.

utils/                Sharding helpers, halo-exchange helpers, memory utils.

scripts/validation/   Per-figure run + plot drivers (NOT regression tests).
  runner.py           Config-overrides driver
  plots/*.py          One file per figure in the README

tests/                Pytest regression suite (Nx <= 64, fast).
  test_regressions.py Sod-1D + Sedov-3D + Gresho HLLC vs L-HLLC

legacy_submit/        Old SLURM scripts kept for reference, not for use.
```

## Time-step data flow

```
Each iteration of the runner_distributed time loop:

    1. compute_dt_fn(U)      <-- hydro/muscl_hancock indirectly, via integrators.py
       returns dt_cfl as a JAX scalar. float(...) is the host sync point.
       Piggy-backs as NaN check (NaN dt_cfl => U has NaN).

    2. SAVE U_pre + OU forcing state    <-- for the auto-CFL-halving retry

    3. step_fn(U, dt)        <-- JIT'd MUSCL-Hancock:
                                 cons_to_prim -> halo exchange of W ->
                                 muscl_hancock_divergence(W_pad, dt) ->
                                 U_new = U + dt * dU
       Halo exchange is via utils/halo_exchange_auto, sharded by NamedSharding.

    4. Operator-split source terms (each is {apply: fn(U, dt) -> U_new}):
         forcing_state["apply"](U, a, key, dt)    <-- OU forcing
         _isothermal_project(U, cs2_iso, gamma)   <-- soft thermostat
         ext_grav_state["apply"](U, dt)           <-- M82 disk gravity
         cc85_state["apply"](U, dt)               <-- central mass/energy injection
         apply_cooling(U, dt, cfg, solver)         <-- Townsend cooling

    5. Re-check via compute_dt(U_new). On NaN: revert U to U_pre, halve dt, retry.
       Up to MAX_HALVINGS=8 (dt -> dt/256). After 8 failed retries:
       dump emergency checkpoint, raise RuntimeError, SLURM gets exit code 1.

    6. Periodic diagnostics (every cfg.diag_interval):
       single GPU reduction -> min(rho), min(P), max(|v|), max(M), sum(rho), sum(E)
       print to stdout + append row to <output_dir>/history.csv

    7. Snapshot (every cfg.snapshot_times entry):
       each process writes its addressable shard via write_snapshot_distributed.

    8. Wall-time checkpoint (every cfg.checkpoint_wall_time_interval seconds):
       write_checkpoint_distributed, same per-shard pattern.
```

## Sharding pattern (multi-GPU / multi-node)

Two rules to follow when adding sharded auxiliary arrays:

1. **Build local-slice only, assemble via `make_array_from_single_device_arrays`** — the KH IC pattern. Naive full-array `device_put` deadlocks multi-process.
2. **Pass sharded arrays as JIT arguments, not Python closures** — XLA tries to inline closure-captured sharded arrays as MLIR constants and crashes with "Fetching value for jax.Array that spans non-addressable devices".

Both rules are documented in `~/.claude/projects/-raven-u-uli-jax-hydro/memory/feedback_sharding_pattern.md` and `feedback_jit_closure_sharded.md`.

## Adding a new source term

1. Write `setup/source_X.py` with `setup_X(cfg, sharding, is_main_process) -> dict | None`. Returns `{"apply": fn(U, dt) -> U}` or `None` when disabled.
2. Call it in `setup/runner_distributed.py` at the source-term setup block (currently lines ~73-81): `x_state = setup_X(cfg, sharding, is_main_process)`.
3. Call `x_state["apply"](U, dt)` inside `_attempt_step` next to the other source-term applies.
4. Add config knobs (e.g., `cfg.x_enabled`, `cfg.x_param1`) to `template_config.py`.

That's the entire contract. See `setup/source_cc85.py` for the canonical reference.

## Adding a new IC

1. Write `ic_generation/<name>.py` with a function `<name>(cfg, x_slice=None, y_slice=None, z_slice=None)` returning the conservative state `U` of shape `(Nvar, Nx_local, Ny_local, Nz_local)` for the local shard.
2. Register in `ic_generation/ic.py`'s `_IC_REGISTRY` with a unique test_problem string.
3. If the problem supports the well-balanced integrator, also write an `_equilibrium` variant and register it in `_EQ_REGISTRY`.

## Adding a regression test

1. Append to `tests/test_regressions.py` a function `test_<short_name>()`.
2. Use `_run_short(cfg)` helper to evolve and `assert` on physical properties.
3. Keep Nx <= 64 so the test runs in single-digit seconds on a laptop CPU.
4. Run locally: `pytest tests/test_regressions.py -v`.
