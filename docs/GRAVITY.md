# Gravity: the multigrid solve and its knobs

Each step solves lap(psi) = source once (`luna/gravity/kdk.py`, `backend.py`). The first solve,
and the first one after a restart, is a cold full multigrid (FMG). Every later step runs
V(1,1) cycles on the potential carried from the previous step (the *warm* solve). By default
they run as in RAMSES (Guillet & Teyssier 2011): from a seed extrapolated in time, until the
rms residual is below 1e-4 of the rms source, at most 10 cycles, each an FP32 cycle on the FP64
residual (`gravity_warm_tol=1e-4`, `gravity_warm_max_cycles=10`, `gravity_warm_extrapolate`,
`gravity_proper_mixed`). With `gravity_warm_tol=0` the solve runs `gravity_warm_vcycles`
(proper time) or `dm_warm_vcycles` (comoving) fixed cycles, the earlier default. This page is about how that solve is set up and what it costs. `docs/CONFIG.md` lists
every field.

## The sharded-solve knobs

| field | default | measured alternative | what it does |
|---|---|---|---|
| `gravity_mg_agg_n` | 128 (was 32) | 32 | On a device mesh the shards coarsen until the global grid side is about this, then every GPU gathers that root grid and solves it on its own copy. |
| `POISSON_N_COARSEST` | 50 | None | Red-black Gauss-Seidel sweeps on the coarsest grid. None means 4 when that grid is the whole periodic grid with side <= 4, and 50 otherwise. |
| `gravity_mg_ca` | `"auto"` (= `"0"` since 2026-10-11) | `"1"` | `"1"` turns on the communication-avoiding smoother: one wide halo exchange per smoothing pass instead of one per sweep. `"auto"` used to turn it on with more than one process. |

`gravity_mg_agg_n` became 128 and `gravity_mg_ca="auto"` became off on 2026-10-11 (A100 numbers
below); `POISSON_N_COARSEST` is unchanged. The MI300A alternatives below were measured on the code before the gathered root
grid (one device per multigrid level count) and are not re-measured since.
All three leave the convergence factor of a V-cycle unchanged. Measured from zero on a
random zero-mean source, rms residual reduction per call: 0.1854 for one V(1,1) and 0.0427
for two, at 1, 2 and 8 GPUs, before and after.

### Measured effect

MI300A, ROCm 7.2, jax 0.10.0, `AMD_SERIALIZE_COPY=3`, command buffers off. 512³ per GPU
(1 GPU 512³ dense solver; 2 GPUs 1024×512² on (2,1,1); 8 GPUs 1024³ on (2,2,2), 4 nodes).
"Op" is the warm solve alone (`backend.solve`, median of 20 calls). "Step" is
`gravity.solve` inside the driven self-gravitating box (`tools/step_breakdown.py`, median of
steps 4-13). Both runs use the gas runs' two V(1,1) cycles, FP64. Times are in ms.

| | defaults (agg 32, 50 sweeps, CA auto) | alternatives (agg 128, None, CA 0) | change |
|---|---|---|---|
| op, 1 GPU | 21.00 | 20.19 | −4 % |
| op, 2 GPUs | 28.44 | 24.87 | −13 % |
| op, 8 GPUs | 44.26 | 37.40 | −16 % |
| step, 1 GPU | 21.41 | 20.39 | −5 % |
| step, 2 GPUs | 28.64 | 25.04 | −13 % |
| step, 8 GPUs | 47.60 | 38.94 | −18 % |
| cold FMG op, 8 GPUs | 56.52 | 50.67 | −10 % |

The cold FMG's error against a converged solve is unchanged (4.62e-3 at 8 GPUs, 5.63e-3 at
1 GPU). One V(1,1) cycle at 64 GPUs (2048³, (4,4,4), 32 nodes, profiled run): 28.2 ms with the
old defaults and 23.7 ms with the new ones.

A100 (Raven, jax 0.10.1, 2026-10-11; gravity audit).
The production warm solve (FP32 V(1,1) on the FP64 residual, to 1e-4, 3 cycles), median ms,
one process per GPU, in-place halo, CA off:

| | agg 32 | agg 128 | change |
|---|---|---|---|
| 256³ per GPU, 2 GPUs | 9.74 | 9.05 | −7 % |
| 256³ per GPU, 4 GPUs | 11.74 | 10.59 | −10 % |
| 512³, 2 GPUs | 28.84 | 26.48 | −8 % |
| 512³, 4 GPUs | 17.70 | 16.86 | −5 % |

The old halo gave the same picture: −6..−11 % at 2 and 4 GPUs. agg 8 was 1.6x slower, and agg 256
+9 % (512³, 2 GPUs). The convergence (3 cycles, residual 1.75e-5) is the same; bitwise equality across
agg_n was not checked under `--xla_gpu_deterministic_ops`. Not yet measured with agg 128 at 8 GPUs
or beyond.

### Why the alternatives

* **Gather at 128.** Below a local side of about 64 a halo exchange costs the same 0.1 ms
  whatever its size, because it is latency-bound. Gathering the 128³ grid removes those
  levels, and the gather itself costs at most 0.3 ms. Gathering earlier, at 256, costs more
  than it saves (+2.3 ms per cycle at 8 GPUs).
* **4 coarsest sweeps on a periodic 4³ grid.** On that grid every mode except the mean is
  damped by about 2/3 per Jacobi sweep, so 4 red-black sweeps do what 50 did. The 50-sweep
  loop cost 1.3 ms per cycle (1 GPU and 2 GPUs, traced), the 4-sweep loop 0.1-0.2 ms.
  Elsewhere 4 sweeps are not enough, which is why the rule falls back to 50. Measured factor,
  128³: Neumann boundaries 0.478 with 4 sweeps against 0.293 with 50; an elongated periodic
  grid (512×128², coarsest 16×4×4) 0.80 against 0.20.
* **CA off.** On the A100 (2026-10-11, production warm solve, in-place halo, one process per GPU)
  CA costs +17 % (2 GPUs, 9.74 against 11.39 ms) and +10 % (4 GPUs, 11.74 against 12.95) at 256³
  per GPU, and +12..24 % at 512³. With the old concatenating halo it was −4 % within a node and
  +2 % on 8 GPUs across two nodes (27.06 against 27.62 ms); the cold FMG (2,2) gained 16-21 %.
  CA on and off give the same bits (shard-0 checksums, 2 and 4 GPUs). The CODE_NOTES gain
  (−24/−29 % at 4/8 GPUs) was MI300A with the old halo and is not re-measured with the new one.
  On MI300A, CA lost at every scale measured: +0.6 ms per cycle at 2 GPUs, +2.2 ms at 8 and
  +1.9 ms at 64 (traced). It halves the exchanges on the coarse levels. But at 512 and 256 it
  doubles the bytes sent, builds a wide-halo copy of the grid, and smooths a larger frame.
  The old "auto" turned CA on whenever there was more than one process, which included
  2 GPUs on one node. Counting nodes instead would still have turned it on where it lost.

The cold FMG on a gathered ladder now runs a whole FMG on the gathered grid instead of one
V-cycle from zero, so gathering more levels does not loosen the cold solve.

### When to revisit

* **Beyond 64 GPUs.** The exchange count grows with the mesh (81 RCCL kernels per cycle at
  8 GPUs, 129 at 64). CA on the coarse levels only, or gathering at 256 again, may win there.
  Re-run the knob scan (`tools/mg_profile/`, branch `mg-profile`) at the new scale before
  changing anything.
* **A different per-GPU size.** With 256³ per GPU the fine level is 8× cheaper, so the
  latency-bound levels weigh more. `gravity_mg_agg_n` is a side length, so it gathers the
  same global grid whatever the per-GPU size.
* **Non-periodic boundaries** keep 50 coarsest sweeps. The knobs above were timed only on
  periodic boxes.

## Testing the warm solve

`tests/test_gravity_warm*.py` run the warm solve along a sequence of sources and compare
each step with a converged solve (FMG + 30 V(2,2) cycles) of the same source. They score the
potential and the acceleration (rms error over rms reference) and assert both stay below a
per-case bound. The bound is 1.5 times the largest error that the fixed two-cycle solve (one
mixed-precision cycle for cosmo) reaches on that case. Each case also runs a deliberately
broken solve: it skips one step's solve (for the restart case, the solve after the restart). The
same bound must catch it, so every test is shown to fail on a broken solve, whatever the solve
setting.

| case | file | what changes from step to step |
|---|---|---|
| jeans | `test_gravity_warm.py` | a linear Jeans mode growing as exp(t) |
| dt_jump | | a moving over-density; dt jumps x4, then drops x0.1 |
| restart | | turbulence; the history is lost halfway (a cold solve, as after a restart) |
| turbulence_fast / _slow | | a log-normal Ornstein-Uhlenbeck density field, dt = 0.05 / 0.005 correlation times |
| collapse_periodic / _isolated | `test_gravity_warm_events.py` | homologous collapse of a uniform sphere to a density contrast of ~500 |
| sinks | | a contracting clump with its core mass removed three times |
| feedback | | turbulence plus a sudden mass injection that then spreads |
| cosmo | `test_gravity_warm_cosmo.py` | comoving gas + CIC dark matter on a growing Zel'dovich field, mixed precision |
| sharded | `test_gravity_warm_sharded.py` | the turbulence case on a (2,2,2) mesh, gathered or not, equal to one device to 1e-10 |

The CPU versions use 32³ and take about 2 minutes together. The `gpu`-marked versions use
128³ and skip on the CPU backend:
`JAX_PLATFORMS=rocm pytest -m gpu tests/test_gravity_warm*.py`. To run the suite on another
solve setting, set `LUNA_TEST_GRAVITY="field=value+field=value"`; it overrides the config
of every case.

`LUNA_TEST_GRAVITY_BOUNDS=tight` selects proposed bounds (`TIGHT` in `tests/_gravity_warm.py`)
for the extrapolated, convergence-controlled, mixed solve: 1.5 times its own largest error, on
the warm steps only (cold steps keep the committed bounds). With them every case passes under
that solve and fails under the fixed two cycles, except jeans. They were a proposal for tol 3e-3, which was not adopted.

## The convergence-controlled warm solve (default)

| field | default | what it does |
|---|---|---|
| `gravity_warm_tol` | 0 | > 0: run V(1,1) cycles until the rms residual is at most this times the rms source, at least one and at most `gravity_warm_max_cycles`. 0 keeps the fixed `gravity_warm_vcycles` (`dm_warm_vcycles` comoving). |
| `gravity_warm_max_cycles` | 8 | the cap on those cycles |
| `gravity_warm_extrapolate` | False | seed the solve with psi_n + q (psi_n − psi_n−1), q = dt_n / dt_n−1, instead of psi_n |
| `gravity_warm_dt_ratio_max` | 4 | extrapolate only while 1/4 <= q <= 4 |

This is what RAMSES's multigrid does (Guillet & Teyssier 2011): iterate until the residual is
small against the source. The test is one global reduction per cycle, inside the jitted solve
(a `while_loop`), so the step keeps a fixed structure. The number of cycles and the final
relative residual go into the step's stats (`mg_cycles`, `mg_residual`).

The extrapolated seed falls back to the carried psi when its history is not valid: the first
solve after the start or a restart (psi is not checkpointed), a dt ratio outside the range,
and the solve two steps after a sink is created or merged, while psi_n − psi_n−1 spans that
sudden mass change. Keeping psi_n−1 costs one
extra grid field per GPU (1 GiB at 512³ FP64). Runs that differentiate through the solve keep
the fixed cycles (a `while_loop` has no reverse-mode derivative). Isolated and periodic
boundaries, on one GPU or a mesh, take the same loop; on a mesh the root grid is the gathered one.

### Choosing the tolerance

The tolerance was chosen on the warm-solve suite (32³ CPU, 128³ GPU) and on the turbulent
cloud with sinks (`scripts/coupling/tier1/luna_cloud_rad.py`, 256³, 1 GPU, 0 to 4 free-fall
times, about 2000 steps). The criterion: every case's largest error is at most that of
today's fixed two cycles. With the extrapolated seed, 3e-3 is the largest value that passes
everywhere. 5e-3 passes the suite but not the cloud (largest sampled acceleration error
1.22e-3 against 8.71e-4). 1e-2 fails the Jeans case at 128³ (6.3e-3 against 1.0e-3).

Suite, 128³, largest acceleration error over the run and cycles per step (measured):

| case | fixed 2 cycles (today) | tol 3e-3, extrapolated | cycles, mean (range) | fixed 1 cycle, extrapolated |
|---|---|---|---|---|
| jeans | 9.96e-4 | 9.71e-4 | 1.07 (1-2) | 6.28e-3 |
| dt_jump | 6.03e-3 | 1.34e-3 | 2.00 (1-3) | 2.92e-2 |
| restart | 1.05e-2 | 1.58e-3 | 3.90 (3-4) | 6.06e-2 |
| turbulence_fast | 1.56e-2 | 1.10e-3 | 4.00 | 9.84e-2 |
| turbulence_slow | 4.90e-3 | 1.52e-3 | 3.00 | 3.18e-2 |
| collapse_periodic | 4.12e-3 | 8.12e-4 | 2.59 (2-3) | 1.86e-2 |
| collapse_isolated | 3.26e-3 | 6.11e-4 | 2.62 (2-3) | 1.42e-2 |
| sinks | 2.30e-3 | 5.16e-4 | 2.21 (2-3) | 1.09e-2 |
| feedback | 1.09e-2 | 1.69e-3 | 3.82 (3-4) | 5.73e-2 |
| cosmo (one mixed cycle today) | 1.74e-2 | 1.38e-3 | 1.65 (1-3) | 1.42e-2 |

The suite's turbulence cases advance the whole field by 0.5 % and 5 % of a correlation time
per step, and take 3-4 cycles; the real runs below take 1.0-1.2. One fixed cycle from the extrapolated seed is 4-10 times worse than today in every
case except cosmo.

Turbulent cloud with sinks, 256³, 1 GPU. "Converged" is tol 1e-5 (4.2 cycles per step). The
error is sampled every 25th step. Solve time is the median per step.

| setting | solve ms | cycles | accel. error median / max | 32nd sink at t | M_sink at t = 3 / 4 |
|---|---|---|---|---|---|
| converged (tol 1e-5) | 6.90 | 4.21 | 9.1e-7 / 5.0e-6 | 1.4859 | 1656.2 / 1842.3 |
| fixed 2 (today) | 3.11 | 2 | 3.1e-4 / 8.7e-4 | 1.4325 | 1640.9 / 1858.1 |
| tol 3e-3, extrapolated | 2.10 | 1.15 | 1.8e-4 / 7.8e-4 | 1.4859 | 1655.6 / 1839.3 |
| tol 5e-3, extrapolated | 2.02 | 1.06 | 2.5e-4 / 1.2e-3 | 1.4304 | 1640.5 / 1853.5 |
| tol 3e-3, carried seed | 3.39 | 2.01 | 3.1e-4 / 5.5e-4 | 1.4325 | 1640.9 / 1856.2 |
| fixed 1, extrapolated | 1.81 | 1 | 2.7e-4 / 5.1e-3 | 1.4326 | 1640.7 / 1857.4 |

The first 16 sinks form at the same times in every run, to 1e-4. Times are in code units;
the free-fall time is 1.07. With tol 3e-3 the cycles were 1 on 1718 steps, 2 on 253 and 3 on
27. The guard fell back to the carried seed on 33 of 1998 steps: the first solve and the
second solve after each step that created or merged sinks. The tol 5e-3 and converged rows
predate a fix of the guard for sink events on consecutive steps (20 fallbacks instead of 33).

Driven self-gravitating box, 512³, 1 GPU, 2500 steps: the solve takes 20.5 ms per step today
and 12.6 ms with tol 3e-3 (1.11 cycles per step in the first 100 steps, 1.00 after). The
acceleration error drops from 1-2e-4 to about 6e-6 once the turbulence is developed.
On 8 GPUs (1024³, 1032 steps) it takes 38.9 ms today and 23.1 ms with tol 3e-3 (the same
cycle counts). A step breakdown over steps 4-13 alone, timed part by part, measured 44.1 ms
against 39.0 ms; it did not record the cycle counts of those steps.

The check costs one extra pass over the fine grid per cycle: the residual norm takes
1.45 ms at 512³ on one GPU, and the source norm 0.38 ms once per solve. One adaptive cycle
takes 11.8 ms against 9.7 ms for one fixed cycle. On 8 GPUs (1024³) it takes 21.9 ms against
18.4 ms, because the residual also needs a halo exchange and a psum. The extrapolation is a
separate pass: 1.0 ms at 512³.

### Re-measured on the gathered root grid

After the root-grid change the options were measured again on two runs at 256³:
the turbulent cloud with sinks and isolated gravity (`luna_cloud.py` at 256³, BOUNDARY=4,
POISSON_BOUNDARY=3, barotropic, 0 to 4 t_ff, about 1455 steps, 32 sinks) and the Evrard
collapse in a periodic box (adiabatic, minmod, t = 0 to 1.5, 555 steps). Errors: max over
the run of |g − g_ref| / max |g_ref| in the densest 1 % of cells, sampled every 4th (cloud)
or 5th (Evrard) step; g_ref is a solve of the same source to an rms residual of 1e-11.
Solve time: `gravity.solve` from the step breakdown, mean over the run (MI300A).

| setting | cloud ms (1 / 2 GPUs) | cloud error | Evrard ms (1 / 2 GPUs) | Evrard error | cycles cloud / Evrard |
|---|---|---|---|---|---|
| 2 fixed V(1,1), FP64 (earlier default) | 4.25 / 6.07 | 1.6e-2 | 3.77 / 6.89 | 2.1e-3 | 2 / 2 |
| 4 fixed | 8.18 / 12.41 | 4.4e-4 | 7.36 / 13.32 | 8.6e-5 | 4 / 4 |
| mixed, 2 fixed | 3.62 / 5.65 | 1.6e-2 | 2.70 / 6.85 | 2.1e-3 | 2 / 2 |
| tol 3e-3, extrapolated | 3.94 / 7.45 | 4.2e-3 | 2.41 / 4.24 | 5.0e-3 | 1.53 / 1.00 |
| tol 3e-3, extrapolated, mixed | 3.14 / 5.40 | 4.2e-3 | 1.97 / 4.10 | 5.0e-3 | 1.53 / 1.00 |
| tol 1e-4, max 10, extrapolated | 7.35 / 15.11 | 3.0e-4 | 3.25 / 5.92 | 1.2e-4 | 3.18 / 1.43 |
| tol 1e-4, max 10, extrapolated, mixed (default) | 5.47 / 9.48 | 3.0e-4 | 2.51 / 5.46 | 1.2e-4 | 3.18 / 1.43 |

The default is the last row: RAMSES's epsilon 1e-4 and cap of 10 cycles, with mixed cycles.
Against the earlier default it cuts the densest-cell error 52× (cloud) and 17× (Evrard); the
total energy stays within 2e-4 (cloud) and 2e-5 (Evrard) of |W0| of a run solved to 1e-9,
against 1.4e-2 and 1.7e-3 before. Tol 3e-3 was rejected: its Evrard error (5.0e-3) is worse
than the fixed two cycles'. Mixed cycles change no error by more than the third digit. With isolated faces on two GPUs
an adaptive FP64 cycle costs about 1.6 ms more than a fixed one; this is not explained.

## Mixed precision for proper-time gravity (default)

`gravity_proper_mixed=True` runs the proper-time warm cycles in FP32 on the FP64 residual of
the carried psi, under the FP64 carry: the same scheme the comoving runs use
(`gravity_precision='mixed'`). The opening cold solve stays FP64. With isolated boundaries the
multipole face values enter the FP64 residual only; the FP32 correction has zero face values.
The table below predates the gathered root grid and isolated mixed cycles:

| | today (FP64, 2 cycles) | mixed, 2 cycles | tol 3e-3, extrapolated | mixed + tol 3e-3 |
|---|---|---|---|---|
| suite 128³: largest error, all 10 cases | as above | identical to 3 digits | as above | identical to 3 digits |
| cloud 256³: accel. error max / M_sink at t = 4 | 8.7e-4 / 1858.1 | 8.7e-4 / 1858.1 | 7.8e-4 / 1839.3 | 7.8e-4 / 1839.3 |
| cloud: solve ms | 3.20 | 2.21 | 2.10 | 1.75 |
| driven 512³, 1 GPU: solve ms | 20.5 | 12.8 | 12.6 | 10.1 |
| driven 1024³, 8 GPUs: solve ms | 38.9 | 28.6 | 23.1 | 17.7 |

The sink history and the cycle counts of the mixed runs match the FP64 runs step for step. One
warm cycle on the op (warm seed, ms): 9.8 → 7.1 at 1 GPU, 12.0 → 10.8 at 2 GPUs, 18.4 → 15.2 at
8 GPUs. The sharded gain is smaller because the FP64 residual on a mesh is a separate
materialized pass.

## Energy coupling: `gravity_energy` ('auto' default, 'kick', 'flux')

The default `'auto'` selects `'flux'` wherever it runs (`setup/gravity_energy.py:flux_requirements`): dense
3-D PLM gas on a proper clock, with or without dark matter, XLA or Pallas hydro. Everything else selects
`'kick'`; startup reports the resolved mode. An explicit `'flux'` refuses what it cannot run
(`flux_requirements`); what it runs but 'auto' does not pick yet is `flux_not_default`: comoving runs (below). The
A7/A8 kick baselines set `'kick'` explicitly.

The step is kick-drift-kick (`luna/gravity/kdk.py`): a half-kick with g^n, the hydro step, the solve for
psi^n+1, a half-kick with g^n+1. The momentum kick is rho g with the centred difference of psi
(`backend.kick_gas_body`). With `'kick'` the energy gets the kick's own work, so the kick leaves the
internal energy unchanged. The total E_kin + E_th + W is then not conserved (A8: |dE/E0| 3.0 at 32^3 and
0.58 at 256^3 by t = 3).

With `'flux'` (Mullen, Hanawa & Gammie 2020, ApJS 252, 30, eq. 57-68) every face's work over the step,

    w_f = -dt / dx F_f (Phi_R - Phi_L),   Phi = (Phi^n + Phi^n+1) / 2,

goes into the energy of the cells beside it. F is the hydro step's own Riemann mass flux.
- Since W = 1/2 sum rho Phi is quadratic with a symmetric Green's function, W^n+1 - W^n = sum (rho^n+1 -
  rho^n) Phi.
- Summation by parts of the continuity update then makes sum_f w_f exactly -(W^n+1 - W^n), however each
  w_f is shared between its two cells.
- The opening half-kick keeps its work W1, so the hydro sees an unchanged internal energy. The closing
  half-kick changes the momentum as 'kick' does, bit for bit, and the energy by the face work minus W1
  (`backend.close_flux_body`).

AthenaK's `SelfGravity` is **not** this coupling: it uses a single-time Phi (`srcterms.cpp:207-213` says
so), so its energy error, 3.7e-6 at 256^3 in A8, is truncation converging at third order, not a tolerance.

**Where the face work goes.** The work w_f is split half to each side, as Mullen et al. eq. 57 do. The
exception is a face next to a cell that the half split would leave with less than `FLUX_ETH_KEEP` (0.5) of its
internal energy: there w_f goes to the donor cell, the one the mass leaves. The half split alone puts half
the potential energy of mass a cell did not hold into the cell beside it. Next to the Evrard cloud edge
(ambient u = 0.05, rho 2.4e-4) that is more than the ambient's internal energy, and the gas goes to p < 0
(NaN at the first step at 32^3, at the bounce at 64^3). Donor-cell everywhere runs without a floor, but it
places the heat at first order: it doubles the Evrard profile error. A conservative limiter backs the
switch up: where a cell would still drop below the same fraction, it keeps that fraction, and the energy
kept back is taken from all cells in proportion to their internal energy (two global sums). With nothing
flagged, both the donor pass and the limiter are skipped (`lax.cond`), and the result is the half split
bit for bit. `stats["flux_moved"]` is the energy the limiter moved.

Evrard (periodic L = 4R, hllc-minmod, FP64 on an A100, **no floor**; jobs 31175590/1, CPU 31175600):

| N | \|dE/E0\| t = 1 / 3 | limiter moved / \|E0\| | L1 \|d ln rho\| vs AthenaK PLM 256^3, t = 1 / 3 |
|---|---|---|---|
| 32 | 3.7e-5 / 1.1e-6 | 2.5e-4 (5 steps) | 0.19 / 0.28 |
| 64 | 7.4e-6 / 1.1e-5 | 0 | 0.122 / 0.104 |
| 128 | 3.3e-5 / 1.7e-5 | 0 | 0.046 / 0.049 |
| 256 | 1.7e-5 / 1.2e-5 | 0 | 0.0116 / 0.0067 |

- Kick: \|dE/E0\| 1.31 / 3.02, 0.73 / 1.55, 0.39 / 1.01, 0.20 / 0.58, and profile L1 at t = 3 of 0.73, 0.40,
  0.21 (64-256). The kick's core
  re-expands.
- Profile L1 at t = 1 / 3 for the other splits:
  - half / half alone (with a floor): 0.126 / 0.102 and 0.046 / 0.049;
  - donor-cell everywhere: 0.148 / 0.234 and 0.066 / 0.112, at 64 and 128.
- The internal energy stays positive: min e_th / e_th(0) never goes below 0.45.
- hllc-mc at 64^3 still dies at t = 2.24. The hydro step itself goes to p < 0 in the box corner, where the
  outflows of neighbouring images collide (the closing kick changes e_th there by < 1 %). Kick at 32^3 dies
  too (A8 xfail).

What "conserved" means: a discrete invariant, E_kin + E_th + W with W from the 7-point operator the solve
inverts, conserved per step to the solve's tolerance.
- Evrard 32^3: max \|dE/E0\| to t = 1.5 is 4.4e-5 at tol 1e-4 and 4.4e-11 at tol 1e-10.
- Per step at 64^3: 6.6e-6 at tol 1e-4, 9e-10 at 1e-8 (critic).
- Measured with the spectral Green's function instead, the energy error is the Poisson truncation:
  3.3e-2, 1.1e-2 and 3.5e-3 at 64-256 (critic).
- The strong case for flux is the solution's accuracy against an independent code (the table above), not the
  invariant.
- A warm solve that stops at `gravity_warm_max_cycles` above `gravity_warm_tol` is refused (`Gravity.check`).
- The opening solve is brought to the warm tolerance in this mode (one extra warm solve at start).

**Smooth flow.**
- A7 (Jeans) is unchanged by the split to 4 digits.
- The growing mode's 2-4 % lower L1 comes from that extra opening solve, not from the coupling: kick with the
  same opening solve matches flux to 9 digits (critic).
- On the Jeans energy exchange, flux's temporal error constant is 1.8x kick's, at the same (second) order.
- Momentum: the same kick in both modes. The periodic drift is the solve's residual (turbulent box 64^3:
  \|dP\| / sum \|rho v\| 1.1e-8 at tol 1e-4).

**Isolated boundaries** (BOUNDARY 4, POISSON_BOUNDARY 3, 64^3, W from LUNA's own solve; job 31175596). The
~1e-4 is mostly energy carried through the box faces, not the multipole Green's function:
- the diode lets mass in from t = 0 (3.8e-5 of it by t = 0.25), a hydro BC issue of its own;
- booking it per step (the hydro's boundary energy flux, plus F_rho Phi at the ghost on every boundary face)
  brings \|dE/E0\| at t = 1 from 1.4e-4 to 1.4e-5 (tol 1e-4), and 2.1e-5 at tol 1e-8;
- the rest (~1e-5, independent of tolerance) is the Green's function's asymmetry (8.9e-6, critic);
- booked, it is 8.6e-5 at t = 2 and 2.7e-4 at t = 3, after the outflow (unbooked 0.07, 0.20).

**Dark matter** (`run='gas+dm'`). The particles keep their leapfrog. The gas's face work uses the total
potential, so the gas's share of W changes exactly. The particles' own budget, dK_dm + sum (rho_dm^n+1 -
rho_dm^n) Phi dV, is not closed.
- Turbulent box with one particle per cell, 64^3 to t = 0.75 (jobs 31175967/8): \|dE/E0\| 1.35e-4 in flux mode,
  all of it the particles' (the rest 2e-9); 4.1e-4 in kick mode.
- It does not shrink with dt (CFL 0.8 to 0.4: 1.349e-4 to 1.350e-4). It is spatial: the PM force (4th-order
  gradient, CIC-gathered) is not the gradient of the CIC-interpolated potential. With the 2nd-order gradient
  it is 8.0e-4.
- Closing it needs an energy-conserving PM force (the derivative of the interpolated potential, Hockney &
  Eastwood), which gives up momentum conservation (self-force). Not done.

**Pallas hydro.** The Pallas kernels (fused and split paths, every dimension) return the face mass fluxes,
bitwise equal to the XLA step's. Each cell stores its lower and upper face, so inner faces are written twice
with identical bits. `hydro_kernel='auto'` takes Pallas on a GPU in flux mode too.

**What it needs, and what it refuses.** The face mass fluxes come from `Hydro.step_flux`: N_d + 1 faces per
shard along d, so a seam face is computed on both sides and no exchange is needed. psi gets a one-cell halo,
as for the gradient. Refused (`flux_requirements`, `Gravity`):
- ORDER 0, `hydro_fused`, AMR, comoving runs;
- sinks;
- chemistry scalars;
- and, because they change mass outside the fluxes, the SiL corrector (`sil_weights`), `cosmo_rho_floor`
  and the cc85 wind.

`gravity_kernel='pallas'` runs the closing kick as a Pallas kernel, bitwise equal. Autodiff runs it in XLA
(VJP: transpose to round-off; the step against central differences, as the kick's).

**Cost** (A100, FP64, median step). The non-flux Pallas step is unchanged in bits and speed: 13.53 / 13.54 ms
at 256^3 and 44.06 / 44.05 ms at 384^3, main / branch on one node (job 31175913). The flux rows are job 31176511.
| | 256^3 | 384^3 |
|---|---|---|
| hydro step, Pallas, Mcell/s without / with the mass fluxes | 1186 / 1147 (+3.4 %) | 1268 / 1238 (+2.5 %) |
| hydro step, XLA | 741 / 733 | 753 / 751 |

Whole step, 256^3 turbulent box, FP64 solve, Pallas gravity kick (job 31176512):
- kick with Pallas hydro: 21.3 ms, 4.87 GiB;
- flux with Pallas hydro: 26.3 ms, 5.80 GiB (+23 %, +19 %);
- flux with XLA hydro: 33.8 ms.

Before the Pallas fluxes, flux cost +47 % against the fastest kick. The rest of the gap is the closing kick's
face-work and positivity passes and the opening kick's stored work.

`tests/test_gravity_energy_flux.py`.

### Comoving flux (explicit only)

Super-comoving gas (Martel & Shapiro 1998) feels Phi~ = s psi, s = (3/2) Omega_m H0^2 a, lap psi = delta. With
W~ = 1/2 s sum (rho~ - <rho~>) psi dV and psi linear in rho~,

    W~1 - W~0 = sum (rho~1 - rho~0) sbar psibar dV + (ds/2)(W~0/s0 + W~1/s1),   sbar, psibar the step averages,

so the face work on Phi = sbar psibar (`backend.step_potential`, static on the solve mode) leaves
d(E~ + W~) = (ds/2)(W~0/s0 + W~1/s1) per step: the trapezoid of Layzer-Irvine's W~ d ln a. The proper-time
(s0 psi0 + s1 psi1)/2 misses it by O(ds dpsi). The comoving-energy stage (gamma != 5/3) and the floors change
only e_th and are booked. The DM leapfrog is unchanged.

Measured (A100 FP64, `tests/test_gravity_energy_flux_comoving.py`, `luna_tests/flux_p0p1/`): the top-hat at 64^3
to turnaround, per-step ledger max |res|/|W~| 3.1e-5 at the default tol 1e-4 and 2.6e-11 at 1e-10 (kick 2.4e-3);
the old Phibar 1.7e-5 (32^3); gamma 1.4 2.1e-11; gas + DM pancake: gas part 1.6e-11 (kick 3.2e-3).

Not the default, because of cold gas. The face work minus the kick's change of K~ lands in e_th, which in the
cold IGM is ~1e-10 of K~ (top-hat, P0 = 1e-10) or 2e-4 (pancake, cs 1 % of v~):
- top-hat 64^3: the positivity limiter moves 3.4 % of |W~|; the IGM ends at 12x the floor's e_th (kick 1.8x);
- Zel'dovich pancake at a = 0.75 (half the shell-crossing a), |T a^2 / K rho~^(2/3) - 1| L1: flux 1246, 408, 38
  at 16, 32, 64 against kick 275, 75, 19 (both converge; the total-energy e_th of a cold flow is truncation).
Default comoving flux needs a dual-energy or entropy fallback first.

Groundwork for flux beyond the dense grid (P0): `close_flux(..., phi_ext)` adds a cell-centred, step-averaged
external potential to the face potential; `Gravity.pre_solve` ops (flux mode) change the mass before the closing
solve and are booked, `W_booked = sum drho Phibar dV` (mean drho removed when periodic) and `E_booked` in the stats,
always present in flux mode; `evolve.step` clears the step's mass fluxes at its end.

## Tried and not kept

**Overlapping the halo exchange with interior smoothing.** One red-black sweep at 512³ per
GPU was split into a local stencil over the whole block (zero ghosts, no communication) and a
face correction from the six exchanged ghost planes. The two forms agree to 7e-16. Measured
per sweep:

| | plain sweep | split, faces by in-place updates | split, faces in one pass | interior only (no face fix) |
|---|---|---|---|---|
| 2 GPUs, (2,1,1) | 3.55 ms | 9.26 ms | 4.25 ms | 3.73 ms |
| 8 GPUs, (2,2,2) | 5.20 ms | 10.9-11.3 ms | 6.28 ms | 5.70 ms |

Even the interior-only form, which skips the face work, is slower than the plain sweep on
8 GPUs. With the latency-hiding scheduler on, XLA schedules the interior kernel between
the start and the end of the exchange, but the time does not drop. On this stack (ROCm 7.2,
jax 0.10.0) the exchange does not run concurrently with the stencil kernel, which matches the
0.01 ms of overlap the profile measured. In-place face updates cost one full-grid pass each.

**A fused Pallas (Triton) red-black sweep.** Each tile recomputes the red values its black
cells need, so a whole sweep reads phi and the source once and writes once. It matches the
XLA sweep to 7e-16. Measured at 512³ on one GPU, best tile 1×8×256: the kernel alone takes
3.22 ms against 3.24 ms for the XLA sweep, and the two-cell padded input it reads costs
another 1.31 ms. Smaller and larger tiles were slower (3.27-5.18 ms). That is far from the
1.3× on the fine level required to keep it.

## Final state, measured

The August ladder harness (`scripts/validation/runs/gravity_blocks_roofline.py`, `step_warm`:
the PM gravity step of the dark-matter runs, one warm V(1,1), 512³ per GPU, jax 0.10.2, ≥ 20
reps). Each tree was timed in the same allocation, in ms:

| GPUs | August | dfac017 (before) | this branch |
|---|---|---|---|
| 1 | 58.3 | 60.95 | 58.28 |
| 8 | 72.0 | 73.60 | 70.11 |
| 64 | 79.3 | 79.56 | 76.15 |

One earlier 64-GPU run of this branch measured 145.6 ms. The same step on the same 32 nodes
an hour later measured 76.15, and the branch with the old agg_n=32 measured 77.24 (with the
old CA on as well: 79.20). The outlier is not explained and did not come back.

The gas runs' warm solve (`backend.solve`, 512³ per GPU, FP64, median of 20 calls), in ms:

| GPUs | old knobs (agg 32, 50 sweeps, CA on) | default now (2 cycles) | mixed, 2 cycles | 1 adaptive cycle, extrapolated | same, mixed |
|---|---|---|---|---|---|
| 1 | 21.00 (before) | 20.16 | 12.88 | 12.80 | 10.03 |
| 8 | 45.43 | 38.28 | 30.94 | 23.18 | 17.62 |
| 64 | 51.47 | 45.51 | 35.63 | 27.49 | 21.32 |

The 1-GPU old-knobs entry is from the dfac017 tree: the 1-GPU solve is the dense one, where
the knobs only change the coarsest sweeps. The adaptive columns time a solve that stopped
after one cycle, which is what the real runs did on 86-100 % of their steps.
