# Differentiating LUNA

What differentiates, what holds it honest, what it costs, and what is known
to be broken. Measured on one MI300A (viper, ROCm 7.2, jax 0.10.0, FP64)
on 2026-09-21; the table and the figure under `docs/bench/` are the output of
`tools/bench_autodiff.py` and can be regenerated with it. The long-horizon
section was measured on jax 0.10.1 on 2026-09-27 (`docs/bench/revolve/`).

## What differentiates

`luna/autodiff/rollout.py:build_simulate` returns

    simulate(theta, U0) -> U_final

which scans the SAME step the production loop takes — `luna/setup/evolve.py`'s
palindrome of stages around the hydro — in a fixed-length `lax.scan` with the
body under `jax.checkpoint`. There is no second, differentiable copy of the
physics to drift from the first. It differentiates in both the in-loop source
parameters `theta` and the initial state `U0`, forward and reverse, on one
device and on several: reverse mode transposes straight through the
`shard_map` collectives. `autodiff_checkpoint` picks what the reverse pass
keeps: `"step"` (the default above) one state per step, `"revolve"` a fixed
number of states -- see "Long horizons" below.

Inside it:

- **Hydro.** Plain AD through MUSCL-Hancock. The limiter and the Riemann
  solver make the step piecewise differentiable, which is a property, not a
  bug — see the identity tests.
- **Gravity.** The FMG Poisson solve is differentiated by an implicit adjoint
  (`luna/gravity/poisson_diff.py`), not by unrolling the V-cycles. The
  gradient of the solve is the solve of the gradient, so its cost and its
  memory do not grow with the cycle count.
  Limits: for the isolated boundary (3) the adjoint freezes how the multipole
  face values depend on rho, so the gradient is approximate in the far-field
  boundary; the sharded isolated solve has no implicit adjoint, so autodiff
  runs periodic; `solve_poisson_fmg_bidiff` does not support boundary 3.
- **Radiation.** The M1 stage (`luna/sources/radiation.py`) differentiates end
  to end, gas and radiation together. Its half-step is recomputed in the
  backward pass rather than stored (`rad_remat`, default on). The transport
  itself is validated in 3D as well as in 1D --
  `tests/test_radiation_m1_3d.py`, see below.
- **dt.** `autodiff_dt_mode="stopgrad"` (the default) hides the CFL estimate
  from the tangent, so no tangent reaches the reduction collectives; `"diff"`
  differentiates through dt and then AD and finite differences agree, kinks
  and all.

Take every gradient of `simulate` under `jax.jit`: an uncompiled reverse
pass materialises the tape unoptimised and segfaults past a few hundred steps
(seen on both ROCm stacks).

## What holds it honest

Three test files, all in the fast tier or just above it:

- `tests/test_autodiff_identities.py` — the checks that do not need a
  reference value: the transpose (dot-product) identity for the hydro step
  and for both Poisson boundaries, gradient identities that follow from
  conservation (the gradient of total mass through a gravity step, of total
  energy), linearity of the Poisson solve, the chain rule over two steps, and
  the two places where the step is only piecewise differentiable.
- `tests/test_autodiff_validation.py` — agreement with finite differences
  (measured floor 6.2e-7), one-sided differences through a shock, the custom
  VJP against unrolled AD, and fp32 input rounding. It also contains a teeth
  check: a deliberately corrupted adjoint must fail the identities.
- `tests/test_autodiff_sharded.py` — the same hydro gradient with the grid
  split over 2 and 4 devices, so every halo crosses a device seam: see
  "Across a shard seam" below.
- `tests/test_autodiff_stability.py` — the tangent-propagator tools
  (`luna/autodiff/stability.py`) against dense linear algebra, minmod's
  derivative at flat and generic states, and the opt-in smooth tangent: see
  "Linear stability" below.
- `tests/test_poisson_neumann_matrix.py` — builds the discrete operator as a
  matrix and compares the implicit adjoint against an explicit pseudo-inverse.
- `tests/test_radiation_m1_3d.py` — the radiation transport in three
  dimensions, which conservation cannot see: a beam streams along each axis
  at c with no leakage into the other two, a diagonal beam keeps its
  direction (the off-diagonal pressure terms) AND its local flux stays at
  |F| = cE, the diffusion limit spreads at 2Dt with an anisotropy below
  1e-6, and an isotropic packet stays invariant under all three axis
  transpositions to 1e-10.

Those 3D gates were checked against broken code, which is the only reason to
trust them: an axis-swapped pressure row, a dropped off-diagonal and a
flipped flux difference. The dropped off-diagonal initially passed all six —
the energy centroid barely feels it, because the energy is advected by F_x
while it is F_y that quietly decays — so the diagonal test now looks at the
local flux, and the break is caught.

The Neumann case deserves a note, because it looked like a solver bug for a
day and was not one. The pure-Neumann Laplacian is singular: its null space is
the constants, and it only solves sources that sum to zero. A one-voxel
finite-difference bump breaks that condition, so the difference between the
adjoint and the finite difference came out as a constant field — the null
space itself, 7.74e-5 at every voxel. Probing along `e_i - mean(e_i)` keeps
the probe inside the solvable subspace, and the agreement is then 6.5e-8 to
1.1e-5. The adjoint was right; the question was inadmissible.

## What it costs

`docs/bench/autodiff_bench.md` is the table, `autodiff_bench.json` the same
rows machine-readably, `autodiff_bench.png` the figure. One MI300A, FP64,
Sedov-Blast-3D, periodic, fixed dt, three timed repetitions after the compile.

The numbers that matter:

| | grad / forward | notes |
|---|---|---|
| hydro, 64^3 | 2.3 | launch-bound at this size |
| hydro, 128^3 | 5.3 | flat in rollout length, 1 to 16 steps |
| hydro, 256^3 | 5.4 | flat |
| gravity, 64-128^3 | 2.8-3.0 | the implicit adjoint, flat |
| radiation, 128^3 | 4.2-5.6 | rises with rollout length |
| coupled, 64^3 | 2.8-3.5 | |

- **Memory grows slowly with rollout length, and fast with resolution.** The
  scan stores one carry per step and the body is remat'd, so 16 steps of hydro
  at 128^3 peak at 3.74 GiB against 2.41 GiB for one step. The state itself is
  84 MiB.
- **The longest hydro rollout that still differentiates at 128^3 is 255
  steps** (bisected: 255 runs, 256 does not), at 22.4 GiB. That limit is
  probably not memory; see "Long horizons" for the crash that revolve avoids.
- **Two GPUs halve both.** The gradient at 256^3, 8 steps: 0.577 s and 24.6
  GiB on one device, 0.289 s and 12.4 GiB on two — 2.00x, with the peak
  memory halved, single process, two local devices.
- **Radiation is the expensive one.** At 128^3 its gradient peaks at 20.6 GiB
  against 3.1 GiB for the hydro alone, and the coupled gradient at 256^3 does
  not fit at all: XLA asks for 133 GiB against a 104 GiB limit and
  rematerialization cannot get below 98.7 GiB.
- **Rematerializing the radiation half-step** (`rad_remat`, default on) cuts
  that peak by a third for 9-11% more time: at 128^3, 30.6 -> 20.6 GiB
  (2 steps) and 30.9 -> 20.8 GiB (4 steps); at 64^3, 4.03 -> 2.51 GiB. At a
  single step it is slightly worse (17.4 -> 18.1 GiB), because there is
  nothing yet to save.

## Long horizons: binomial checkpointing

`autodiff_checkpoint="revolve"` (`luna/time_integration/revolve.py`) keeps
`autodiff_snapshots` states instead of one per step, or as many as
`autodiff_snapshot_gib` holds per device, and recomputes the rest on the
binomial schedule (Griewank 1992; Griewank & Walther 2000 -- the offline
form of revolve, since n is fixed). With s snapshots, each of n steps is
advanced at most t times, t the smallest integer with C(s+t, t) >= n, and the
number of steps advanced is the optimum, t n - C(s+t, t-1).

How it runs. The schedule is built on the host as integer tables. A
`custom_vjp` stores the snapshots in the forward sweep. The backward pass is
one device loop over the n reversed steps: restore a snapshot, advance and
store as the table says, take the VJP of one step. The program does not grow
with n. The step is the production step (shard_map halos, implicit-adjoint
FMG). Its VJP is the one per-step checkpointing takes, so the gradient is
the same numbers: **bitwise identical** to per-step checkpointing, measured
for hydro and hydro + self-gravity, s = 1, 3, 10 of 10 steps. That holds on
CPU (16^3, one device and a 2x2x2 mesh) and on the MI300A (32^3, one and two
GPUs). Reverse mode only; `autodiff_grad_mode="forward"` falls back to
"step".

**What it costs, measured** (`docs/bench/revolve/`; one MI300A unless the
GPU column says 2; FP64, minmod, Sedov-Blast-3D, fixed dt; jax 0.10.1;
"advanced / n" is the schedule's recomputation, forward sweep included):

| case | grid | GPUs | steps | checkpoints | grad s | grad / fwd | advanced / n | peak GiB |
|---|---|---|---|---|---|---|---|---|
| hydro | 128^3 | 1 | 120 | every step | 1.10 | 5.59 | 1.00 | 11.36 |
| hydro | 128^3 | 1 | 120 | revolve 16 | 1.26 | 6.35 | 1.86 | 3.15 |
| hydro | 128^3 | 1 | 250 | every step | **SIGSEGV** | | | |
| hydro | 128^3 | 1 | 1000 | revolve 8 | 14.56 | 8.82 | 4.29 | 2.52 |
| hydro | 128^3 | 1 | 1000 | revolve 16 | 12.19 | 7.36 | 2.86 | 3.15 |
| hydro | 128^3 | 1 | 1000 | revolve 64 | 10.66 | 6.45 | 1.94 | 6.98 |
| hydro | 128^3 | 1 | 2000 | revolve 16 | 26.24 | 7.93 | 3.43 | 3.15 |
| hydro | 128^3 | 1 | 2000 | revolve 64 | 21.45 | 6.48 | 1.97 | 6.98 |
| gravity | 128^3 | 1 | 16 | every step | 0.29 | 4.10 | 1.00 | 5.17 |
| gravity | 128^3 | 1 | 32 | every step | **SIGSEGV** | | | |
| gravity | 128^3 | 1 | 1000 | revolve 16 | 21.53 | 6.70 | 2.86 | 5.31 |
| gravity | 128^3 | 1 | 2000 | revolve 64 | 37.35 | 5.80 | 1.97 | 12.06 |
| hydro | 256^3 | 1 | 32 | every step | 2.27 | 5.24 | 1.00 | 36.12 |
| hydro | 256^3 | 1 | 120 | every step | **OOM** (88 GiB asked) | | | |
| hydro | 256^3 | 1 | 1000 | revolve 32 | 87.87 | 6.49 | 2.41 | 36.07 |
| hydro | 256^3 | 1 | 2000 | revolve 32 | 183.82 | 6.80 | 2.70 | 36.07 |
| hydro | 256^3 | 2 | 2000 | revolve 32 | 92.19 | 7.07 | 2.70 | 18.06 |
| gravity | 256^3 | 1 | 8 | every step | 0.93 | 4.51 | 1.00 | 33.25 |
| gravity | 256^3 | 1 | 16 | every step | **SIGSEGV** | | | |
| gravity | 256^3 | 1 | 1000 | revolve 32 | 132.42 | 6.63 | 2.41 | 61.89 |
| gravity | 256^3 | 2 | 1000 | revolve 32 | 80.70 | 6.57 | 2.41 | 31.23 |
| gravity | 256^3 | 2 | 2000 | revolve 32 | 167.22 | 6.86 | 2.70 | 31.23 |

All 62 rows, 120 to 2000 steps in every combination, are in
`docs/bench/revolve/autodiff_bench.md`.

- **Memory is flat in n.** Each snapshot costs one state: 82 MiB per slot
  at 128^3 hydro (the state is 84 MiB), 1.16 GiB at 256^3 with gravity (U,
  psi, g: 1.13 GiB). The rest is one step's tape. Two GPUs halve it.
- **The time follows the schedule.** grad/fwd = r + (advanced/n - 1), where
  r is the per-step ratio. Subtracting the recomputation gives r = 5.49-5.51
  over every hydro 128^3 row (5.59 measured per step at 120 steps), 5.09-5.11
  at hydro 256^3 (5.24 measured at 32 steps), and 5.15-5.22 at gravity 256^3.
  Choose s by memory: going from 16 to 64 snapshots at 2000 steps buys
  7.93 -> 6.48.
- **The longest horizon shown at 256^3 is 2000 steps**, hydro + FMG
  self-gravity, 167 s per gradient on two MI300A, 31.2 GiB per device.
  Hydro alone: 2000 steps on one MI300A, 184 s, 36.1 GiB.
- **No host or disk level.** Nothing measured here needed one: the largest
  case uses a third of a device.

**Why every loop bound is hidden from XLA.** On this stack (ROCm 7.2, jax
0.10.1, MI300A) a compiled while loop whose trip count XLA can read
segfaults once the trip count is large enough. The same loop with a runtime
bound runs. Measured forward-only, no AD:

| loop | static trip count | runtime trip count |
|---|---|---|
| hydro 128^3, `fori_loop` | 800 runs, 1000 SIGSEGV | 1000 runs |
| hydro 64^3, `fori_loop` | 1000 SIGSEGV | 1000 runs (1.4 s) |
| gravity 128^3, `fori_loop` / `scan` | 120 SIGSEGV | 120 runs (5.0 s) |

The segfault persists with `HIP_LAUNCH_BLOCKING=1`, with command buffers off
and with while-loop unrolling off. It is not memory and not the physics:
the same 1000 steps as ten host-driven 100-step pieces stay finite. The
threshold falls as the loop body grows, so the gravity step hits it sooner.
`revolve.py` passes every bound through `lax.optimization_barrier` and jits
the rollout, because an eager call hands `fori_loop` a concrete bound, and
that is a static trip count again. The eager path broke the IC inversion's
first forward call before it was jitted.

The per-step mode cannot do the same: `scan` needs a static length for its
tape. Its gradient fails on this stack at hydro 128^3 / 250 steps, gravity
128^3 / 32 steps and gravity 256^3 / 16 steps. Those are the black crosses
in the figure. The 255-step "memory limit" above (jax 0.10.0) is probably
the same failure; it was not re-measured on 0.10.0.

**What holds it honest** (`tests/test_autodiff_revolve.py`, CPU):
the schedule reverses every step from the right state, and its advance count
equals the closed form, which equals a brute-force optimum. The gradient
equals per-step checkpointing bitwise (hydro, gravity; 1 device, 2x2x2;
s = 1, 3, 10). The steps executed, counted by a host callback, equal
the schedule's count plus one linearization per step. XLA's temp memory
falls by one state per snapshot. Each was run against a broken copy:

| break | fails | still passes |
|---|---|---|
| suboptimal split (m = 1 always) | 13 schedule + 3 count tests | the gradients (a valid, slow schedule) |
| restore always from slot 0 | all 4 gradient tests | the counts |
| buffer sized by n, not s | the memory test | the gradients |
| budget ignores the device count | the budget test | |

**The 256^3 IC inversion at 1000 steps** (the existing driver,
`scripts/examples/autodiff/ic_inversion_run.py problem=jeans nx=256
n_steps=1000 LIMITER=0 MESH_SHAPE=2,1,1 autodiff_checkpoint=revolve
autodiff_snapshots=32`, one node, two MI300A, jax 0.10.1). The rollout is
8.3x longer than the 120-step production figure. On one GPU the per-step
gravity gradient at 256^3 already fails at 16 steps on this stack. The FD
gate at the centre voxel agrees to 1.1e-5. 150 Adam iterations at 80.9 s
each (3.4 h) bring the residual ||delta - delta_true|| / ||delta_true|| from
0.98 to **0.0226**. r(k) is 1.00 in every shell from k = 3 to the grid scale,
0.99 at k = 2 and 0.86 at k = 1. That k = 1 shell holds only a few modes, so
its sample is the smallest. The residual equals the 120-step figure's 0.02,
at 8.3x the horizon. Figure: `docs/bench/revolve/ic_inversion_jeans_256_1000steps.png`.

## What is known broken

**Differentiated radiation on ROCm needs serialized kernel launches.** The
compiled reverse-mode program of a radiation rollout of 8 steps or more dies
with SIGSEGV on its second or third execution -- not on the first, and not in
the forward pass, which runs indefinitely. The diagnosis, in the order it was
measured (2026-09-21, MI300A, ROCm 7.2):

- memory is flat and far from the limit when it dies (2.6 GiB of 104.5 at
  64^3, 21.4 of 104.5 at 128^3), with preallocation on and off, so it is
  neither capacity nor the allocator;
- the hydro and gravity gradients of the same length never crash, and neither
  does radiation at 4 steps over six calls;
- it happens on jax 0.10.0 AND on 0.10.1 (on 0.10.1 it usually dies one call
  earlier), so it is not the known 0.10.0 autodiff segfault;
- disabling XLA command buffers saves 64^3 but not 128^3; disabling
  autotuning or the latency-hiding scheduler does not help at all;
- **making the radiation program smaller fixes it with the launches left
  async**: at 128^3 / 8 steps, `rad_substeps=1` survives (0.445 s per
  gradient), `rad_order=1` survives (0.524 s), both together 0.226 s, while
  the default two-substep second-order program dies. Fewer groups does not
  help, and neither does turning the remat off, so it is the size of the
  program and not the amount of memory it keeps;
- the coupled case at 128^3 / 8 steps still dies even with one substep: the
  multigrid's own instructions push it back over whatever the threshold is;
- **`HIP_LAUNCH_BLOCKING=1` fixes all of them**, and so does
  `AMD_SERIALIZE_KERNEL=3`: 6 calls at 64^3 / 8 steps, 3 calls each at
  64^3 / 16, 128^3 / 8 and coupled 128^3 / 8.

So it is an async-dispatch race in the ROCm runtime that a large enough
program triggers, not an autodiff bug, and it is the second time LUNA has hit
one on the adjoint path (the multi-GPU IC inversion needed the same
variable). The forward simulation never meets it: this is reverse mode only,
and hydro-only and gravity-only gradients of any length in the table are
clean and run at full speed.

**What it costs depends entirely on which physics is in the run**, so it is
not a variable to export by default. Gradient seconds, async -> serialized,
three to five timed repetitions each:

| | async | `HIP_LAUNCH_BLOCKING=1` | |
|---|---|---|---|
| radiation 128^3, 4 steps | 0.583 | 0.570 | free |
| radiation 64^3, 4 steps | 0.0881 | 0.0844 | free |
| hydro 64^3, 8 steps | 0.0123 | 0.0122 | free |
| hydro 128^3, 8 steps | 0.0764 | 0.0760 | free |
| hydro 256^3, 8 steps | 0.5786 | 0.5786 | free |
| **gravity 128^3, 8 steps** | **0.509** | **1.569** | **3.1x slower** |

The multigrid is a long chain of small kernels and it is exactly what
serialized launches punish; `AMD_SERIALIZE_KERNEL=3` is no better there
(1.762 s). Turning command buffers off costs nothing anywhere measured
(gravity 0.546 -> 0.560 s, hydro 0.0764 -> 0.0759, radiation 0.576 -> 0.571)
but only saves 64^3.

So there is no single switch to recommend:

| what you are differentiating | what to do |
|---|---|
| hydro, gravity, or both | nothing: full speed, no crash at any length here |
| radiation, rollout <= 4 steps | nothing |
| radiation at 64^3, longer | `XLA_FLAGS=--xla_gpu_enable_command_buffer=` (free) |
| radiation at 128^3, longer | fewer `rad_substeps`, or `HIP_LAUNCH_BLOCKING=1` |
| radiation + gravity, longer | `HIP_LAUNCH_BLOCKING=1`, and the multigrid pays 3x |
| anything, on more than one node | never `HIP_LAUNCH_BLOCKING`: it hangs the first cross-node collective ("Across nodes") |

`setup_radiation` prints this warning when a differentiable run starts on
ROCm without any of it, and says nothing when there is no radiation to
protect. The real fix is to make the radiation transport emit fewer
instructions -- the axis loop, the RK2 stages and the substep scan multiply
out -- which would speed the gradient up as well as move the threshold; that
is the next piece of work, not a flag.

**The stack was not the pinned stack, and the pinned one is better.**
`pyproject.toml` pins jax 0.10.1; the venv every earlier run used,
`venv_jax_rocm7.2`, is 0.10.0 -- the version the audit records
as segfaulting in AD. A sibling venv was built to settle it,
`venv_jax_rocm7.2_j1011` (same packages, jax and the ROCm
plugin at 0.10.1, without mpi4py/mpi4jax). On 0.10.1:

| | 0.10.0 (installed) | 0.10.1 (pinned) |
|---|---|---|
| `tests/test_sedov_gradients.py` | SIGSEGV | passes, 70 s |
| `tests/test_cosmo_hydro_music_blocks.py` | 17 s | 21 s (0.10.2: > 2700 s) |
| fast tier | 28 pass, 76 s | 28 pass, 84 s |
| fast tier, `--devices 2` | 27 pass | 27 pass |
| the three autodiff + radiation files | pass | pass |

So the pin is right and the environment is wrong: switch the venv to 0.10.1
(or use the sibling), and the forward-mode Sedov gradient test comes back from
the dead. 0.10.2 remains disqualified by the 160x cosmology regression.

**Restart with radiation is refused.** A checkpoint carries U and the OU
forcing state, not the radiation field, so a restart would silently begin
again from `rad_E0`. `Config.validate()` raises rather than let that happen.

**Sinks and prescribed-orbit particles are refused** in `autodiff_mode`:
their per-step state is not threaded through the scan carry.

**Forward mode** through the sharded step is refused where a `shard_map`
collective has no JVP rule; reverse mode is the supported direction there.

**A benchmark is not a proof of scaling.** Everything above is one node. The
1-vs-2-device numbers are single process, two local devices, not the
multi-process path the production runs use.

## Across a shard seam

`tests/test_autodiff_sharded.py` runs the production MUSCL-Hancock step
(minmod, HLLC, the wide 2-ghost halo, XLA kernel) and a 4-step checkpointed
rollout of it on 16^3, FP64, with the grid on (2,1,1), (2,2,1) and (4,1,1)
device meshes. With two devices on a periodic axis the left and the right
neighbour are the same device, so a halo sent the wrong way round is
invisible there; the 4-device meshes are what can see it. Measured
2026-09-26:

| | 8 virtual CPU devices | 2 MI300A (jax 0.10.0) |
|---|---|---|
| `<Jv,w>` vs `<v,J^T w>`, one step | 3.4e-15 - 3.5e-15 | 1.8e-16 |
| the same, 4-step rollout | 2.0e-15 - 2.8e-15 | 2.2e-16 |
| d(total mass)/dU - (1,0,0,0,0) | 1.6e-17 | 9.1e-18 |
| gradient, N devices vs 1 | 4.0e-16 | 2.7e-16 |
| J v, N devices vs 1 | 0 - 1.2e-16 | 0 (bitwise) |
| J^T w, N devices vs 1 | 1.1e-16 - 1.7e-16 | 1.1e-16 |

So the sharded derivatives are the single-device ones to round-off; the
tangent is often bitwise equal, the adjoint differs in the last bit (the
seam cotangents are summed in another order). The assertions sit at the
identity tests' 1e-11. On two GPUs the (2,2,1) and (4,1,1) cases skip.

**Finite differences at the seam run at a generic state.** The identity
tests' `_smooth_state` has flat directions (v_z = 0, v_x uniform in x and
y): there minmod(a, b) sits at a = b = 0, where it has no derivative (see
"Linear stability"), and a central difference floors at 1e-7 - 3e-5 per
cell, identically on one device and on two -- the limiter, not the seam.
`generic_state` varies every primitive in every direction. There, with the
loss <w, U(T)> for a fixed random w, the best central difference over
h = 1e-3 .. 1e-6 agrees with the gradient to 5.6e-10 - 5.7e-8 at seven seam
cells (every variable, both sides of the interior seam and the periodic
wrap), identically on one and two devices; the test asserts the validation
file's 1e-6. The 1/2<U,U> loss is not used there: its value is 6.7e3, and
its rounding over h is 3e-6 of the 0.024 gradient of a momentum cell, the
floor it measured on one device and on two.

Checked against broken code (CPU, 8 devices, patches applied to a copy of
the tree):

- the halo exchange's adjoint drops what crosses a device boundary: the
  mass identity fails (7.1e-3 - 7.5e-3), every seam finite difference fails
  (5.9e-3 to 1.6), and the transpose and N-vs-1 tests error out (their JVP
  cannot pass the `custom_vjp` the break uses). The in-file negative
  control makes the same corruption with the tangent taken from the correct
  code, and there the transpose identity and the N-vs-1 comparison fail as
  numbers;
- the forward sends the wrong edge slab across a real seam: the mass
  identity and the N-vs-1 comparison fail on all three meshes, the
  transpose identity and the finite differences pass -- correctly, the
  derivative of the wrong function is still its derivative.

## Linear stability from the tangent propagator

`luna/autodiff/stability.py` treats the rollout as a map U0 -> U(T) and its
Jacobian M as an operator it never forms:

- `leading_mode`: subspace iteration on M with JVPs only (forward mode) --
  the leading eigenvalues, so for a steady base state the modal growth rate
  ln|lambda|/T;
- `optimal_perturbation`: subspace iteration on M^T M, one JVP and one VJP
  per vector -- the leading singular values, the largest energy gain any
  initial perturbation reaches over T;
- both in the Chu (1965) energy norm, rho|v'|^2 + p(rho'/rho)^2 +
  p/(gamma-1)(p'/p - rho'/rho)^2, through explicit linear maps between dU
  and norm coordinates;
- `nonlinear_growth` and `tangent_vs_difference`: seed the NONLINEAR
  rollout with a mode and compare -- the check that the linearisation means
  something.

The states are jit ARGUMENTS and every global sum is a shard_map psum, so
the same calls run on one device or on a sharded mesh. A block of vectors
with Rayleigh-Ritz (default 4) rather than one: symmetric base states have
clustered leading modes -- here two layers times two phases -- and a single
vector crept for 60 iterations without converging.

### The test case

`tools/kh_stability.py` (minmod throughout): a periodic double tanh shear layer, streams at
+-0.5, uniform rho = 1 and p (sound speed 2: Mach 0.25 per stream), layer
thickness d = 0.4446/(2 pi) = 0.0708 so that Lx = 1 carries Michalke's most
unstable wavenumber, Ly = 2 (the layers 14 d apart). gamma = 5/3, HLLC,
CFL 0.4, fixed dt. The base state is an exact steady state of the discrete
scheme (drift over tau: 0.0).

The reference is not a textbook number but the linear theory OF THIS
SETUP: the linearised compressible Euler equations about the same u(y),
Fourier collocation in y, dense eigenvalues (`linear_theory`, numpy). It
reproduces Michalke's 0.1897 for an isolated incompressible layer (0.18970)
and gives, the same to 7 digits for ny = 256, 512, 1024:

| | sigma |
|---|---|
| this setup, the two modes (layers in / out of phase) | 1.257385, 1.240578 |
| the same double layer at Mach 0.005 | 1.346301 |
| Michalke, one incompressible layer, in these units | 1.340418 |

Compressibility at Mach 0.25 costs 6.6% of the growth rate, the second
layer 0.4%. Comparing to the incompressible 0.1897 would have hidden a 7%
effect in "numerical dissipation".

### The finding: minmod's tangent at a flat state is the first-order scheme

A parallel shear flow is flat in x, and flat in y in rho, p and v_y. There
every minmod(a, b) sits at a = b = 0, where it has no derivative: the
directional derivative is minmod(da, db), not linear in the direction, and
`jnp.where(a*b > 0, ..., 0)` takes the zero branch. The JVP of the
production step at such a state is therefore the JVP of the FIRST-order
Godunov scheme -- with its low-Mach dissipation ~c dx -- while the
nonlinear step reconstructs the perturbation at second order. MC and van
Leer behave the same way. Measured on the KH case with plain minmod, the
tangent's growth rate is 0.5787, 0.9069, 1.0786, 1.1671 at 64 x 128 ..
512 x 1024 (-54, -28, -14, -7.2% against theory: first order), while the
nonlinear minmod code seeded at 1e-6 per cell grows at 1.2268 .. 1.2571
(-2.4 .. -0.03%). `test_minmod_tangent_is_the_derivative_only_off_the_crease`
pins it on a 16^3 4-step rollout: at a generic state a one-sided difference
converges to the JVP (1.2e-8 at h = 1e-8), at the flat shear it stays at
5.2e-2 for every h.

The scheme stays minmod. Two remedies change only what is linearised.

**Route 1 -- a base state off the crease** (`kh_stability.py --base-delta`).
Every primitive of the shear layer gets a small smooth structure: two
waves each, delta x (rho, c_s, c_s, p), kx = 3-5 x 2 pi/Lx (k d = 1.3-2.2,
beyond the KH cut-off, so they do not grow), random ky and phase. No
limiter argument is then exactly zero and no y-face has S_M = 0; the JVP is
the derivative of the production step at that state. It is not a steady
state: its waves move (max |U(tau) - U(0)| = 9.4e-3 at delta = 1e-3,
9.4e-4 at 1e-4), so M(tau) is linearised about a slightly moving base.
The reference remains the linear theory of the tanh profile, which the
structure shifts by O(delta) (measured below: delta 1e-3 -> 1e-4 changes
sigma by 1.4e-4 at 128 x 256, 6e-5 at 256 x 512 relative).

**Route 2 -- a smooth tangent for minmod** (`Config.limiter_tangent_eps`,
default 0 = off; needs autodiff_mode, LIMITER=0, no AMR). The forward is
exact minmod -- bitwise equal to 173f12e's step on 1 and 2 CPU devices and
1 and 2 MI300A, with and without the switch -- and the tangent is that of
a minmod whose sign and min are smoothed over e = eps x (rho, sqrt(p/rho),
p) of the cell (`slopes.smooth_minmod`, a `jax.custom_jvp`). At a = b = 0
that tangent is the central slope (da + db)/2 for every e; with |a|, |b|,
||a| - |b|| >> e it is minmod's to O((e/a)^2). The VJP is the rule's
transpose (checked: 9.7e-15 on 1 and 2 devices). At a generic 16^3 state
the smoothed tangent approaches minmod's own as e shrinks: |J_e v - J v| /
|J v| = 6.6e-2, 3.5e-3, 4.4e-4, and below 1e-6 for e = 1e-2, 1e-4, 1e-6, 1e-8
(asserted: shrinking, and < 1e-6 at 1e-8).

**HLLC's S_M = 0 switch.** Every y-face of the flat shear has contact speed
exactly 0, where HLLC's F*_L / F*_R choice is a second kink; route 2 keeps
it (route 1 is off it). Measured on the 16^3 flat shear along a smooth
direction, JVP vs one-sided difference: minmod 4.05e-3 without and 4.13e-3
with a uniform drift v_y = 0.2 that moves every face off S_M = 0; smoothed
tangent 3.27e-3 vs 3.34e-3. So the limiter, not the Riemann solver, is what
the tangent gets wrong there; the switch is not visible at this level.

### Measured growth rates, both routes

Leading eigenvalue of M(tau = 0.5), block of 4, 50 iterations, minmod.
"nonlinear" = the NONLINEAR code seeded with the tangent's leading mode:
route 1 at 1e-10 per cell on the structured base, route 2 at 1e-6 per cell
on the flat base (late rate over 8 tau). "fair" = |D_h - M z| / |M z| over
one tau, D_h the one-sided difference of the nonlinear code at h per cell.
Theory 1.257385. jax 0.10.1, command buffers off, HIP_LAUNCH_BLOCKING=1 on
1 GPU; 512 x 1024 on 2 GPUs.

Route 1, delta = 1e-3, seed 0:

| nx x ny | tangent sigma | vs theory | nonlinear (1st tau / late) | fair, h = 1e-8 / 1e-6 |
|---|---|---|---|---|
| 64 x 128 | 1.174922 | -6.6% | 1.17492 / 1.1725 | 2.4e-5 / 5.4e-4 |
| 128 x 256 | 1.237447 | -1.59% | 1.23745 / 1.2373 | 9.9e-6 / 3.0e-4 |
| 256 x 512 | 1.252460 | -0.39% | 1.25246 / 1.2516 | 7.9e-6 / 1.3e-4 |
| 512 x 1024 | 1.256176 | -0.096% | 1.25618 / 1.2584 | 5.3e-6 / 7.6e-5 |

Second order in dx (the error drops 4.2x, 4.1x, 4.1x). The tangent IS the
derivative: the first tau of the nonlinear run reproduces |lambda| to all
six printed digits and the one-sided difference agrees to 5e-6 - 2.4e-5 --
but only for perturbations far below the structure: at 1e-6 per cell it is
already 1e-4 off, at h = 1e-12 rounding takes over (1e-3). The late
nonlinear rate differs from the tangent's by up to 2e-3 because the base
moves. Another seed (another structure) gives 1.234023 at 128 and 1.251585
at 256 (-0.28% / -0.07% from seed 0); delta = 1e-4 gives 1.237272 and
1.252383. Seeded at 1e-5 per cell instead of 1e-10 the nonlinear rate at
128 x 256 climbs from 1.2375 to 1.2412 as the perturbation outgrows the
structure and the flat-state dynamics take over.

Route 2, e = 1e-4:

| nx x ny | tangent sigma | vs theory | nonlinear minmod, flat base | tangent - nonlinear | fair, any h |
|---|---|---|---|---|---|
| 64 x 128 | 1.257788 | +0.032% | 1.2233 | 2.8% | 3.7e-2 |
| 128 x 256 | 1.259642 | +0.18% | 1.2517 | 0.63% | 1.1e-2 |
| 256 x 512 | 1.257946 | +0.045% | 1.2560 | 0.15% | 2.8e-3 |
| 512 x 1024 | 1.257522 | +0.011% | 1.2571 | 0.037% | 7.3e-4 |

The same to all printed digits for e = 1e-2, 1e-4 and 1e-6 at 64, 128 and
256: on this base every crease cell is exactly flat, where the tangent is
the central slope whatever e, and the layer's own slopes are far above e.
The tangent is not the derivative of the nonlinear minmod dynamics there --
there is none; "fair" does not change with h, the nonlinear map is
positively homogeneous, not linear -- but the gap closes at second order
(3.4x, 3.8x, 3.9x in "fair", 4x in the growth rate).

Which to use for what:

- Route 1 answers "what is the derivative of the production scheme", and
  its answer is checked by finite differences to 1e-5. Its price: the
  answer belongs to a base that is not the textbook one, it depends on the
  chosen structure at 0.3% (128) - 0.07% (256), and it describes only
  perturbations below that structure.
- Route 2 answers "what does the production scheme do to a small smooth
  perturbation of the textbook base", to within 0.04% at 512 x 1024 and
  converging at second order, independent of e over four decades, with a
  forward that is bitwise the production one. It is a linearisation, not a
  derivative, at the crease.
- Plain minmod's tangent at the flat base answers neither.

Cost of the eigen-iteration (50 iterations x 4 JVPs of tau = 0.5):
256 x 512 (800 steps) 61-69 s on 1 GPU, 512 x 1024 (1600 steps) 497-504 s
on 2 GPUs, the same for both routes.

### Optimal perturbations

128 x 256, block of 4, M^T M iterated to rtol 1e-9 or 30 iterations:

| T | route 2, s_1 (ln s_1 / T) | route 1, s_1 (ln s_1 / T) | plain minmod tangent, s_1 |
|---|---|---|---|
| 0.25 | 1.7994 (2.350) | 1.6976 (2.117) | 1.4123 |
| 0.5 | 2.8891 (2.122) | 2.6255 (1.931) | 1.8357 |
| 1 | 6.0802 (1.805) | 5.4458 (1.695) | 3.2027 |
| 2 | 19.328 (1.481) | 18.264 (1.452) | 8.6324 |
| 3 | 70.023 (1.416) | 64.543 (1.389) | 21.652 |

The best gain over a short horizon is far above the modal one --
ln(s_1)/T is nearly twice sigma at T = 0.25; the classical reading is the
Orr mechanism, a perturbation tilted against the shear first grows by
un-tilting -- and ln(s_1)/T falls towards sigma as T grows. Route 2
between T = 2 and 3: 1.287 per unit time, 2.3% above its sigma at this
grid. Route 1 is 8% lower in s_1 at T = 3, as its sigma is at this
resolution; plain minmod's tangent, the first-order scheme's, gains 3x
less.

### ROCm, again

On one MI300A the 256 x 512 and 512 x 1024 eigen-iterations SIGSEGV before
the first iteration on jax 0.10.0, on 0.10.1, and on 0.10.1 without
command buffers; the 2-GPU runs of the same size do not. A staged probe
shows the pattern of the radiation crash above -- the first call of the
800-step JVP or VJP succeeds, the second or third dies. What survives, 8
calls each at 256 x 512 x 800 steps:

| | JVP | VJP |
|---|---|---|
| jax 0.10.0 | dies at call 2-3 | dies at call 2 |
| 0.10.0, HIP_LAUNCH_BLOCKING=1 | dies | |
| 0.10.0, command buffers off | survives | dies at call 1 |
| 0.10.0, both | survives (0.44 s) | survives (0.59 s) |
| jax 0.10.1 | dies | |
| 0.10.1, command buffers off | survives (0.31 s) | survives (0.32 s) |

and the whole 1-GPU eigen-iteration at 256 x 512 still needed
HIP_LAUNCH_BLOCKING=1 on top. So every GPU number in this section is jax
0.10.1 (`venv_jax_rocm7.2_j1011`) with
`XLA_FLAGS=--xla_gpu_enable_command_buffer=`, plus HIP_LAUNCH_BLOCKING=1
for the 1-GPU runs at T >= 2 and at 256 x 512; the seam table above is jax
0.10.0 (`amuse_env010`), where the 16^3 tests do not crash.

## Forward and reverse mode at scale

`tools/bench_autodiff.py --mode jvp` times the forward-mode derivative (a
JVP along a random tangent) where the default times value_and_grad.
Sedov-3D, hydro only, FP64, XLA kernel (autodiff mode always is), fixed dt,
jax 0.10.1, command buffers off, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`
(the default 0.75 caps a device at 82.5 GiB), 2 timed calls after the
compile. Peak is the device high-water mark, per device on 2 GPUs.

| | grid | steps | forward s | derivative s | ratio | peak GiB |
|---|---|---|---|---|---|---|
| JVP, 1 GPU | 384^3 | 8 | 0.339 | 0.986 | 2.91 | 37.5 |
| JVP, 1 GPU | **448^3** | 1 / 8 | 0.065 / 0.524 | 0.182 / 1.502 | 2.81 / 2.87 | 52.7 / 59.4 |
| JVP, 2 GPUs | **512^3** | 1 / 8 | 0.056 / 0.444 | 0.161 / 1.574 | 2.89 / 3.54 | 39.4 / 44.4 |
| VJP, 1 GPU | 320^3 | 8 | 0.214 | 1.047 | 4.88 | 37.5 |
| VJP, 1 GPU | **384^3** | 1 / 8 | 0.042 / 0.339 | 0.208 / 1.732 | 4.96 / 5.12 | 44.4 / 63.0 |
| VJP, 2 GPUs | **512^3** | 1 | 0.056 | 0.271 | 4.85 | 59.5 |

Bold is the largest that ran. What did not: 512^3 on one GPU (the JVP asks
for one 68.6 GiB buffer, the VJP for 84.4 GiB), 448^3 VJP on one GPU,
640^3 on two (JVP 67 GiB, VJP 89 GiB), 512^3 x 8 steps VJP on two
(71.7 GiB); 768^3 on two timed out at 400 s without a message. With the
platform allocator (`XLA_PYTHON_CLIENT_ALLOCATOR=platform`: no BFC cap, no
peak statistics) 512^3 JVP ran on one GPU (2.2-4.3 s) and 512^3 x 8 steps
VJP on two (4.6 s), both much slower than BFC where BFC fits.

The limit is the step, not the tape: the 1-step JVP at 448^3 peaks at
15.7 states' worth of memory, and the single buffer the failed 512^3 JVP
asked for is 13.7 states. The forward step carries ~25 N^3 temporaries
(the optimization_barrier comment in muscl_hancock.py); its tangent and
adjoint multiply that. A leaner derivative of the step -- per-direction
barriers that survive AD, or a custom rule over the fused kernel -- is what
would move these limits.

## Running it

The experiments are tasks in `scripts/examples/autodiff/`, one file each
(`ic_inversion`, `eta_recovery`, `luna_emergence`,
`corrector_training`), run through their setups there:

    python -m scripts.examples.autodiff.ic_inversion_run problem=jeans nx=32
    python -m scripts.examples.autodiff.m82_autodiff_run nx=64 task=eta_recovery n_iter=60
    python -m scripts.examples.autodiff.blast_emergence_run nx=256 movie=frames/
    python main.py my_run.py autodiff_mode=1 autodiff_task=ic_inversion n_iter=40

A task is `task(init, cfg, physics, is_main, *, <options>)`:
`luna.main.run(cfg, task)` builds the mesh, the ICs and the physics as for any
run and hands them over; `cfg.autodiff_task` names the example `python main.py`
uses. An experiment's own parameters (`n_iter`, the emergence target, the
inverted field, ...) are the task's keyword options, not Config fields: on the
command line of a run script or of `python main.py` they are `name=value` like
a field, and a run script binds its defaults with `functools.partial` (`TASK`).
The benchmark
and the stability analysis:

    python tools/bench_autodiff.py --out bench_out \
        --cases hydro,gravity,radiation,coupled --sizes 64,128 \
        --rollouts 1,2,4,8,16 --reps 3
    python tools/bench_autodiff.py --out bench_out --max-rollout hydro:128
    python tools/bench_autodiff.py --out docs/bench --merge a.json,b.json
    python tools/bench_autodiff.py --out bench_out --cases hydro,gravity \
        --sizes 128,256 --rollouts 120,250,500,1000,2000 \
        --ckpt revolve --snapshots 16,32 --reps 2
    python tools/bench_autodiff.py --out bench_out --cases hydro --sizes 448 \
        --rollouts 1,8 --mode jvp          # forward mode instead of the gradient
    python tools/kh_stability.py --nx 256 --mesh 2,1 --tangent-eps 1e-4 \
        --modes theory,eig,nonlinear,svd --json kh.json

Every case runs in its own process (the device high-water mark is not
resettable, an out-of-memory rollout takes the runtime with it, and a warm
compile cache would hide the compile cost of everything after the first
case). On an APU the job MUST export `XLA_PYTHON_CLIENT_PREALLOCATE=false`:
"device memory" there is the node's RAM, and the default preallocation walks
through the job's cgroup limit as soon as a rollout gets long. The symptom is
a SIGSEGV that reads exactly like an autodiff bug and is not one — which is
how the first run of this benchmark produced four of them.

## Across nodes

Forward and reverse mode run on 4 and 8 MI300A over 2 and 4 nodes, one
process per GPU, and the sharded derivatives are the 1-GPU ones to
round-off. Everything here is MEASURED on viper (jax 0.10.1,
`venv_jax_rocm7.2_j1011`, FP64, minmod `LIMITER=0`, fixed dt), 2026-09-26/27;
job ids in brackets.

### The recipe

    mkdir -p slurm_logs
    sbatch --nodes=4 --export=ALL,MESH=4x2x1 submit/examples/autodiff_multinode/check.sbatch
    sbatch submit/examples/autodiff_multinode/icinv.sbatch        # 256^3 IC inversion, 8 GPUs
    sbatch submit/examples/autodiff_multinode/bench.sbatch        # 512^3 forward/JVP/VJP

`submit/examples/autodiff_multinode/env.sh` holds the flags:

- one task per GPU: `--ntasks-per-node=2 --gres=gpu:2`; both tasks see both
  GPUs and `initialize_jax_distributed` pins each by `SLURM_LOCALID`;
  `JAX_COORDINATOR_ADDRESS` = the first node's IP;
- `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`;
- `XLA_FLAGS=--xla_gpu_enable_command_buffer=`;
- **no `HIP_LAUNCH_BLOCKING`, no `AMD_SERIALIZE_KERNEL`**;
- `AMD_SERIALIZE_COPY=3`: it serializes the runtime's copies, not the kernel
  launches, and closes the crash that `HIP_LAUNCH_BLOCKING` was covering
  (below). `ADJ_SCOPY=0` turns it off;
- `adj_srun`: srun with a fresh coordinator port, relaunched when a step
  dies within a minute (start-up failures, below).

`tools/adj_multinode.py` is the driver behind the scripts: `probe` (every
derivative program called 3 times, a line before and after each call, so a
hang names its program), `ref` + `check` (the correctness gate below, one
`CHECK PASS|FAIL` line), `bench` (time and per-device peak), `coll` (a bare
ring ppermute + psum, no LUNA).

### Why the 8-GPU IC inversion hung: `HIP_LAUNCH_BLOCKING=1`

The July recipe set `HIP_LAUNCH_BLOCKING=1` for every problem. It deadlocks
RCCL across nodes, with no LUNA involved (2 nodes, 4 GPUs, [11991808]):

| program | async launches | `HIP_LAUNCH_BLOCKING=1` |
|---|---|---|
| `coll`: one ring ppermute + psum | 5 calls; 1 ms each after the first | first call never returns (killed at 240 s) |
| Jeans 64^3, 10 steps: forward, VJP, 3 calls each | all return (VJP 0.11 s) | first forward never returns (killed at 420 s) |

Under blocking launches RCCL still initialises (8 `Init COMPLETE` lines,
as with async launches); the log stops while the ppermute's first cross-node
point-to-point channels (`NET/IB`) connect, and the psum is never reached. On
one node the same variable works (the July 2-GPU production figures used
it). The same matrix without it: command buffers on or off, jax 0.10.0 or
0.10.1, the hydro alone, cooling at 64^3 -- all run. Command buffers are off
in the recipe because a replayed shard_map program hangs on one node
(af9d9de) and because of the 1-GPU crash table above, not because they
broke anything across nodes here.

### The crash the variable was hiding: a race in the copies

Without the variable long gradients crash -- on ONE GPU as well, so this is
not a multi-node problem; it is why the variable was set. 1 GPU, 128^3,
cooling problem, VJP called 3-4 times, async launches [11991998, 11992550,
11993031, 11993110, 11993178, 11993231, 11993232]:

| rollout | result |
|---|---|
| 20 steps | SIGSEGV on call 3 once (flat state, also without OU forcing); 4 calls run in the 3 later runs |
| 20 steps, cooling off, or `cooling_n_sub=1`, or 5 steps | runs |
| 60 steps | 4 calls run |
| **120 steps** | **SIGSEGV on call 1, every time** (4 runs; sub-steps as scan or unrolled; no OU: call 4) |
| 120, `HSA_ENABLE_SDMA=0` / `GPU_MAX_HW_QUEUES=1` / `AMD_SERIALIZE_COPY=1` | SIGSEGV on call 1 |
| 120, `AMD_SERIALIZE_COPY=2` or `=3` | 4 calls run, 3.31 s |
| 120, `HIP_LAUNCH_BLOCKING=1` | 4 calls run, 3.62 s |

So it is a race in the runtime's COPIES: waiting for each copy to finish
(`AMD_SERIALIZE_COPY` 2 or 3; 1 waits only before it) closes it and leaves
the kernel launches -- and with them the collectives -- asynchronous. It is
not specific to cooling: the gravity IC inversion at 64^3 x 20 steps on 2
GPUs of one node segfaulted in its first gradient twice without it and ran
with it [11993203, 11993259, 11993260] (the same inversion at 256^3 x 120
steps runs without it, on 2 and on 8 GPUs). Across 2 nodes
`AMD_SERIALIZE_COPY=3` leaves the collectives working: `coll`, the Jeans
probe and the 120-step cooling probe all run [11993054, 11993098]. Cost at
512^3 on 8 GPUs, hydro: forward 0.134 s, JVP 0.356 s, VJP 0.642 s with it,
0.133 / 0.355 / 0.642 s without [11993100, 11992055]; gravity: 32-step VJP
4.966 s with it, 4.900 s without (+1.3%) [11993362, 11992279]; the 256^3
gravity inversion 4.32 s/it with it, 3.99 without (+8%) [11993364, 11992054].

**Start-up failures.** Three times a process segfaulted while the ROCm
client started (after `jax.distributed.initialize` connected, before the
devices were listed), each time with `AMD_SERIALIZE_COPY=3`: 3 of ~18
multi-node launches with it, 0 of ~35 without [11993054, 11993101,
11993362]; each time the next launch in the same job ran. A dedicated count
of 24 bare `coll` launches saw no segfault (3 did not finish within 120 s, 1
of them without the variable; their output was not kept) [11993177]. The
scripts' `adj_srun` relaunches a step that dies within a minute.

Separately, the cooling sub-steps are unrolled when differentiated
(`sources/cooling.py`, `autodiff_mode` only): a speed change, not a crash fix
(the 120-step crash is the same either way). Gradient, 1 GPU: 128^3 x 20
steps 0.545 s unrolled vs 0.758 s as a scan; 128^3 x 120 steps with
`AMD_SERIALIZE_COPY=3` 3.32 vs 4.70 s; 256^3 x 20 steps 4.07 s / 68.3 GiB
unrolled and async vs 5.78 s / 83.6 GiB for the scan with serialized
launches [11992278]. The forward alone is faster as a scan (0.908 vs
1.059 s at 256^3), so plain runs keep the scan.

### Correctness on 4 and 8 GPUs

`check.sbatch`: a 1-GPU reference in the same job (`srun -n1`), then the whole
mesh; 128^3, 20 steps, a generic state (rho, v and p times broadband
perturbations: a density perturbation alone is a static contact mode and the
rollout would not move). Three problems: `hydro` (the Jeans box without
gravity), `jeans` (the gravity IC inversion's physics, FMG adjoint), `cooling`
(OU forcing + radiative cooling). Max |difference| / max |reference|; the
transpose identity with exactly rounded sums (host `math.fsum`; the device
`vdot` adds eps x the dot's condition number, 1.7e3 here, 1e5 on one CPU
case). Tolerance 1e-11 everywhere. 4 GPUs = 2 nodes, mesh 2x2x1 [11992271];
8 GPUs = 4 nodes, 4x2x1 [11992272]:

| | hydro 4 / 8 | jeans 4 / 8 | cooling 4 / 8 |
|---|---|---|---|
| rollout U(T) vs 1 GPU | 0 / 0 | 8.6e-16 / 8.6e-16 | 0 / 0 |
| gradient vs 1 GPU | 1.8e-15 / 2.1e-15 | 2.4e-15 / 2.2e-15 | 7.3e-15 / 7.5e-15 |
| J^T w vs 1 GPU | 7.5e-16 / 7.5e-16 | 9.6e-16 / 1.0e-15 | 1.3e-14 / 1.3e-14 |
| J v vs 1 GPU | 0 / 0 | -- | -- |
| <Jv,w> vs <v,J^T w> | 5.3e-16 / 6.6e-16 | -- | -- |
| d(mass)/dU - (1,0,0,0,0) | 2.4e-16 / 2.4e-16 | 2.2e-16 / 2.2e-16 | 3.3e-16 / 3.3e-16 |

All `CHECK PASS`. The dashes: the gravity solve and the cooling rate are
`custom_vjp` functions, and JAX has no JVP through them, so forward mode --
and with it the transpose identity -- is the hydro's.

`tests/test_autodiff_multiprocess.py` runs the same checks on CPU in 8
PROCESSES of one device each (gloo), 32^3 on 4x1x2, 4 steps; the sharded
tests elsewhere split one process. All pass (e.g. hydro: gradient 8.2e-16,
transpose 4.7e-16; cooling: gradient 5.1e-15). Checked against broken code
(copies of the tree, 8 processes):

- the halo sends its slabs the wrong way round along x: all three problems
  fail (cooling: U(T) 0.12, gradient 0.14, J^T w 0.38; the mass identity
  still holds, correctly -- the wrong scheme is still conservative);
- the halo's adjoint drops the cotangent that crosses a device: all three
  fail (cooling: mass identity 0.15, gradient 0.56, J^T w 0.42; hydro: the
  JVP cannot pass the custom_vjp the break uses).

**The OU forcing was FP32 in FP64 runs.** The first run of the cooling
check, from a flat state at 16^3 on 2x2x2 CPU devices, differed from 1
device by 2.1e-7 (gradient) -- the forcing's wave vectors and cell
coordinates were float32, and a float32 sin/cos rounds differently with the
shard's line length. From a generic state the same break shows as 1.6e-12
against 1.9e-15 in FP64, below the 1e-11 bound, so the multi-process test
cannot guard it; `tests/test_turbulence.py::test_forcing_is_computed_in_the_state_precision`
does (the force against a float64 numpy sum: 5.4e-7 with FP32, passes in
FP64). The forcing now follows the state's precision in `autodiff_mode`
only: in FP64 it costs 26% on a 256^3 driven-turbulence rollout
(0.590 vs 0.469 s, 20 steps, 1 GPU [11992471]), and a plain run does not
need layout-exact forcing.

### At scale

512^3, 8 GPUs on 4 nodes (4x2x1), 64 x 128 x 512 cells per GPU. Median of 2
timed calls after the first; peak is the device high-water mark, the same on
all 8. Measured without `AMD_SERIALIZE_COPY` (with it: hydro the same to 1%,
the gravity VJP +1.3%, see above). `hydro` [11992055, 11992280], `jeans` = hydro + self-gravity through
the FMG adjoint [11992279]:

| | steps | forward s | derivative s | ratio | peak GiB |
|---|---|---|---|---|---|
| hydro, JVP | 8 | 0.133 | 0.355 | 2.67 | 10.1 |
| hydro, VJP | 8 | 0.133 | 0.642 | 4.82 | 19.6 |
| hydro, VJP | 32 | | 2.568 | | 36.8 |
| hydro, VJP | 64 | | 5.137 | | 56.8 |
| hydro, VJP | **112** | | 9.497 | (4.5 est.) | 86.8 |
| jeans, VJP | 8 | 0.423 | 1.450 | 3.43 | 34.2 |
| jeans, VJP | 32 | | 4.900 | | 61.2 |
| jeans, VJP | **64** | | 11.18 | | 97.2 |

Bold: the longest measured. 136 hydro steps died (SIGSEGV), 160 asked for
one 112.7 GiB buffer. The "(4.5 est.)" is 9.497 s over 14 x the 8-step
forward, an estimate. The first call of the two longest rollouts took 643 s
and 485 s (the 32- and 64-step ones 16-50 s): near the memory limit the
first execution is slow, the later ones are not.

The same 512^3 hydro on 2 GPUs of one node (2x1x1 [11991851, 11992443]):
forward 0.450 s (8 steps), JVP 1.461 s (46.9 GiB), VJP 0.276 s for 1 step
(64.5 GiB) and 1.158 s for 4 (71.7 GiB); 8 VJP steps do not fit. So 4x the
GPUs: forward 3.4x faster, JVP 4.1x; the VJP per step 0.080 s against
0.29 s. (The 2-GPU JVP row is from the run on the flat-density state -- the
rerun on the generic state was OOM-killed by the node's cgroup; the kernels
are the same.)

### The 256^3 IC inversions, iterating across nodes

`icinv.sbatch`, the production driver (`ic_inversion_run`), 256^3, 120 steps,
minmod, no serialized launches:

| | GPUs | compile + it 0 | s / iteration | peak GiB |
|---|---|---|---|---|
| gravity (Jeans), 1 node [11991850] | 2 | 85.3 s | 7.35 | n/r |
| gravity (Jeans), 4 nodes [11992054] | 8 | 33.8 s | 3.99 | n/r |
| gravity (Jeans), 4 nodes, `AMD_SERIALIZE_COPY=3` [11993364] | 8 | 33.6 s | 4.32 | 19.9 |
| cooling, 4 nodes, `AMD_SERIALIZE_COPY=3` [11993099] | 8 | 22.6 s | 3.52 | 16.0 |

n/r: not recorded (the runner prints the peak since these runs). The
gravity runs agree iteration by iteration (L = 3.427e-04, residual 0.8078
after 10 iterations on both); 30 iterations on 8 GPUs reach residual 0.497
with r(k) >= 0.91 in the first nine shells. 4x the GPUs make an iteration
1.84x faster. The cooling inversion ran 10 iterations (loss 2.50e-6 ->
1.90e-6); without `AMD_SERIALIZE_COPY` it segfaulted after its first
iteration [11992544]. The July 2-GPU runs (18.9 s/it for gravity) used
another stack and limiter and are not comparable.

### What is still open

- The copy race itself is a ROCm runtime bug that LUNA works around; the
  reproducer is `cooldiag.sbatch` (`CASES=s120`, 1 GPU, 30 s).
- Forward mode through self-gravity and cooling: both are `custom_vjp`.
  The Poisson solve is linear, so a `jax.lax.custom_linear_solve` (the same
  FMG as solve and transpose-solve) would give it a JVP and keep today's
  adjoint; the cooling rate needs a `custom_jvp`. Then the transpose
  identity can be checked through the multigrid too.
- `HIP_LAUNCH_BLOCKING` stays single-node. Whether `AMD_SERIALIZE_COPY`
  also closes the radiation gradient's crash (above) is untested: on this
  branch the differentiable radiation rollout does not trace at all --
  the dark skip in `sources/radiation.py` (`dispatch.run`) calls
  `bool(_quiet(c))` on the traced carry whenever dt is a Python float, as
  autodiff's fixed dt is (TracerBoolConversionError, [11993731, 11993756]).
- Start-up failures (above) are rare and unexplained.

## Solver in the loop at scale

A learned flux correction for 3D driven isothermal turbulence, trained by
reverse mode through long coarse rollouts on several GPUs. The published
solver-in-the-loop work stops at 2D (diffhydro II, arXiv:2512.13403: a
Kelvin-Helmholtz corrector on 128x256) or at 32x32x64 with 16-step unrolls
in 3D (Um et al. 2020, arXiv:2007.00016). LUNA's own 2D example is
`corrector_training` (32^2 against 256^2, docs/GALLERY.md section 4). The
code is `luna/sources/ml_corrector.py` (the network), the `Corrector`
stage and `tools/sil3d.py` (reference, training, baseline, evaluation);
the tests are `tests/test_sil3d.py`.

### The problem

A periodic unit box, rho0 = cs = 1, isothermal through LUNA's projection
(gamma = 5/3 with p = cs^2 rho reset every step), solenoidal OU forcing at
k = 1..2 with correlation time t_turn = L / (2 v_rms). Minmod, FP64. The
reference is LUNA itself at 1024^3, pooled 4x (conservatively: cell means
of rho and rho v) to the 256^3 coarse grid.

The coarse run takes a fixed step dt_c = 0.3 dx_c / (cs (1 + 2 M)). The fine
run steps adaptively and lands exactly on every multiple of dt_c. There it
records the OU mode amplitudes, and the coarse run replays them (stage
`Replay` in `tools/sil3d.py`). Drawing its own, the fine run would drive a
different flow: it takes about four draws per coarse step.

**Calibration, measured** (256^3, 2 GPUs, `tools/sil3d.py ref f=1`). The OU
acceleration is `acc_coef v^2 / L`:

| acc_coef | vfac | rms Mach, 2-5 t_turn | max(|v| + cs) | coarse CFL, max | wall |
|---|---|---|---|---|---|
| 2.0 | 3 | 4.35-5.09 (settling to 4.43 at 3 t_turn) | 11.9 | 0.224 | 77 s, 3 t_turn |
| 2.55 | 2 | 4.97-5.63 | 13.3 | 0.361 | 87 s, 5 t_turn |

The second row is the production setting: Mach ~5 after 1 t_turn, the
coarse CFL below the production 0.4, rho from ~1e-3 to ~2e2.

### The corrector

The network sees ln(rho/rho0), v/cs and ln(p/(rho0 cs^2)) and emits, per cell,
one number o in (-amp, amp) (amp tanh) for each of the 5 fields and 3
directions: a flux through the cell's upper face along that direction,
F = min(rho_i, rho_i+1) cs^n o (n = 1 for mass, 2 for momentum, 3 for
energy). The update is U += -dt/dx sum_d (F_i+1/2 - F_i-1/2), after the hydro
with the full dt, as the innermost stage (`stages.Corrector`, on with
`sil_weights`, off by default). So:

- conservative by construction: every flux leaves one cell and enters its
  neighbour;
- a cell loses at most 6 amp cs dt/dx of its mass (0.16 at the coarse dt
  above): the density stays positive;
- the last layer starts at zero, and a fresh network is the bare solver bit
  for bit;
- sharded: one halo exchange of depth + 1 cells per application
  (`make_halo_shardmap`); the parameters are replicated, and the transpose
  of the shard_map sums their gradient over the shards.

**Dense or separable.** ROCm has no FP64 convolution: MIOpen stops with
"Unsupported DNN data type: tf.float64". A dense 3^3 layer therefore runs
as 9 FP64 matmuls over three stacked z-shifts. The default layer is
separable: a per-channel 3^3 stencil (elementwise, which XLA fuses) and then a
1x1 channel mix. The receptive field is the same, for a fraction of the work.
Measured at 256^3, one MI300A, FP64:

| network | parameters | corrector alone | / bare coarse step (29.9 ms) | K = 16 value + grad |
|---|---|---|---|---|
| dense, width 8, depth 3 | 6,079 | 100.8 ms | 3.37 | out of memory (72 GiB asked) |
| dense, width 16, depth 3 | 15,599 | 164.0 ms | 5.49 | out of memory (90 GiB asked) |
| **separable, width 16, depth 3** | **1,622** | **27.2 ms** | **0.91** | **4.96 s, 60.3 GiB** |
| separable, width 32, depth 4 | 5,526 | 61.9 ms | 2.07 | out of memory (70 GiB asked) |
| separable, width 16, depth 3, 2 GPUs | 1,622 | 14.4 ms | 0.95 (of 15.1 ms) | 2.43 s, 30.5 GiB per GPU |

"Bare coarse step" is the whole production step: hydro, the replayed forcing
and the isothermal projection. The value-and-gradient of the separable
network costs 5.47x its forward, which is the hydro's own ratio (5.3-5.6,
above). The example uses separable, width 16, depth 3.

### What holds it honest

`tests/test_sil3d.py` has 16 tests. They run on the CPU backend with 8
virtual devices (on a GPU the 2x2x2 cases skip), 16^3, a generic state and replayed forcing, and every one
failed on a deliberately broken copy of the code before it was trusted:

| test | broken with | result |
|---|---|---|
| matmul conv == the direct 27-tap sum (numpy) | kernel (j, k) transposed | fails |
| separable == dense conv with W = P D | stencil (i, k) swapped | fails |
| the correction sums to zero per field (both kinds, 1 device and 2x2x2) | left face weighted 0.999 | fails |
| the corrected step keeps the bare step's mass and momentum (forcing off) | same | fails |
| a fresh network is the bare solver bit for bit | +1e-12 on the output | fails |
| 2x2x2 parameter gradient == 1-device, 1e-11 (both kinds) | halo one cell short, edge-padded | fails |
| Taylor remainder falls as e^2, central difference == <g, v> to 1e-6 (every leaf) | 1e-3 of a layer's value hidden from the gradient | fails |
| revolve gradient == per-step checkpointing | wrong snapshot restored | fails |
| sil_weights unset: no corrector stage, the hydro step (alone and fused with the kick) == the raw solver bit for bit (1 device and 2x2x2) | the barrier wrapper off by one ulp; a small random corrector with sil_weights unset | fails, both |

The central difference uses e = 1e-6. At larger steps it crosses minmod,
min(rho) and HLLC kinks. Measured relative error along the test's
direction: +1.9e-5 at e = 3e-4, -2.9e-6 at 1e-4, -1.3e-9 at 1e-6.

The training targets reach every step through `revolve_rollout(aux=True)`.
Every step reads that input, but it is neither differentiated nor stored in
the snapshots.

### The reference

`submit/examples/sil3d/ref.sbatch`: 1024^3, 8 processes on 4 nodes (mesh
2x2x2), seed 7, the fixed coarse step dt_c = 1.065e-4 (256^3, CFL 0.3 at
cs(1 + 2M)). Spin-up to n = 1900 (2.0 t_turn), every coarse step stored to
n = 2900 (the training window, 1.07 t_turn), every 16th to n = 7600
(8.1 t_turn). Measured: 28,730 fine steps, 13,211 s (3.7 h, 14.7
node-hours), 0.27 s per fine step without snapshots; 1,294 pooled
snapshots (float32 rho, rho v), 4.4 s each to write. From 2 t_turn on
(238 diagnostic samples) the rms Mach number is 5.1-6.5, mean 5.6, above the
5.0-5.6 of the same forcing at 256^3 (less numerical dissipation). rho spans
1.4e-3 to 9.3e2. The fine grid's largest |v| + cs is 15.4, which is a CFL of
0.42 at the coarse dt_c (upper bound: the coarse run sees the pooled flow).

### What the reference found: a miscompile at 1024^3

The first reference compiled the whole step (OU kick, hydro, kick,
isothermal projection) as one XLA program, and died at coarse step 107:
rho = -34 in one block, in Mach-0.7 flow. The production loop compiles
each stage on its own and ran clean. `tools/fusediff.py` takes one step
both ways from the same state and compares every cell on every shard.
Measured at 1024^3 on 2x2x2, 8 processes on 4 nodes, (5, 512, 512, 512)
FP64 blocks:

| compiled into one program | vs stage by stage |
|---|---|
| the whole step | every cell differs; momentum up to 4e-4 |
| OU kick + hydro | the same, bit for bit |
| hydro + second kick + projection | exact |
| OU kick + copy / trivial shard_map / halo exchange | exact |
| OU kick + hydro at dt = 0 (an identity) | rho and momentum exact; U[4] wrong in all 134,217,728 cells of every block, by up to 1.1 |
| OU kick, optimization barrier, hydro | exact (<= 6.7e-16) |

At 512^3 on the same mesh the whole step fused is exact (2e-16). Field 4
of a (5, 512, 512, 512) FP64 block starts at exactly 2^32 bytes, which
points at a 32-bit offset in the fused in-place update. That is not proven.
It is the same symptom as the fb_1024 study run's corruption: U[4]
rewritten everywhere, rho and momentum untouched, only at 1024^3.
Rewriting the kick with per-shard coordinates left the differences
unchanged, bit for bit.

The fix is `build_hydro` (luna/setup/hydro.py), which puts the hydro step's input behind
`jax.lax.optimization_barrier`. Called on its own it is the same program as
before. Inlined into a larger program (the autodiff scans, SiL training) it
materialises the preceding stages first. Measured after the fix: the whole
step fused runs to n = 160 with the stage-by-stage run's diagnostics
(Mach 1.116, rho in [0.9940, 1.006]), and the fusediff rows above are exact.
Before the fix, any gradient through `build_simulate` at 1024^3 on 2x2x2
was silently wrong.

**The regression check.** `tools/fusediff.py barrier=0` runs the hydro
without the barrier (the code before the fix) and exits 1 on any differing
cell. Rerun 2026-09-28 on jax 0.10.2 (`jax_env2`), 1024^3, 2x2x2, 8
processes on 4 nodes: barrier=0 gives kick + hydro(dt = 0) U[4] wrong in all
134,217,728 cells of every block (up to 1.1), and kick + hydro and the whole
step different everywhere; barrier=1 gives kick + hydro(dt = 0) and kick +
hydro exact (max |diff| <= 8.9e-16). End to end, `sil3d.py ref fuse=1` to
n = 160: without the barrier it dies at coarse step 107 again (113 cells,
rho = -34, process 7); with it every diagnostic line equals the fuse=0
run's (n = 160: Mach 1.116, rho in [0.9940, 1.006]). (The fusediff "all"
row with the barrier runs out of memory in the compare at 1024^3; the
fuse=1 run is its check.)

**What the barrier costs.** The hydro step alone, one MI300A, FP64, minmod,
Sedov state, raw vs behind the barrier interleaved in one process, median of
20 (the step of
`tools/bench_hydro_roofline.py`):

| grid | kernel | raw | barrier | ratio | fused with an energy update before it: raw / barrier |
|---|---|---|---|---|---|
| 256^3 | Pallas | 6.31 ms | 6.33 ms | 1.002 | 6.44 / 7.33 ms (1.14) |
| 256^3 | XLA | 13.66 ms | 13.67 ms | 1.001 | 13.77 / 14.65 ms (1.06) |
| 512^3 | Pallas | 51.50 ms | 51.51 ms | 1.000 | 51.50 / 59.57 ms (1.16) |
| 512^3 | XLA | 107.21 ms | 107.44 ms | 1.002 | 107.27 / 115.30 ms (1.07) |

Every output is bitwise equal to the raw step's. Called on its own (the
production loop) the barrier costs nothing. Inside a larger program it costs
one pass over U (about 8 ms at 512^3): that pass is the fix, since the state
before the hydro is what it makes XLA write out. Only fused programs pay it:
the autodiff scans and SiL training, which run the XLA kernel.

Weak scaling, the production loop (`scripts/validation/runs/sedov3d_run.py`,
Sedov-3D FP64 minmod, 512^3 per GPU, 60 steps, mean of the perf lines at
steps 40-60 of two runs each, old = the same tree without the barrier,
jax 0.10.2, 2026-09-28):

| GPUs | grid | kernel | old ms/step | new ms/step | new/old |
|---|---|---|---|---|---|
| 1 | 512^3 | Pallas (default) | 52.37 | 52.29 | 0.998 |
| 8 (2x2x2, 4 nodes) | 1024^3 | Pallas (default) | 61.60 | 61.60 | 1.000 |
| 16 (4x2x2, 8 nodes) | 2048x1024^2 | Pallas (default) | 62.73 | 62.75 | 1.000 |
| 1 | 512^3 | XLA | 107.93 | 107.84 | 0.999 |
| 8 (2x2x2, 4 nodes) | 1024^3 | XLA | 115.32 | 115.42 | 1.001 |
| 16 (4x2x2, 8 nodes) | 2048x1024^2 | XLA | 116.16 | 116.11 | 1.000 |

Weak efficiency 1 -> 8 -> 16 GPUs: Pallas 85.0 / 83.5% old, 84.9 / 83.3%
new; XLA 93.6 / 92.9% old, 93.4 / 92.9% new. (The 16-GPU Pallas "old" is
one run: its first launch segfaulted in one process before step 1, the
start-up crash of `AMD_SERIALIZE_COPY=3`.) No narrower barrier is needed.

### Status

Done: the corrector (dense and separable), its stage and config switch
(`sil_weights`, off by default; off, the step is the default solver bit for
bit, `test_sil_off_is_the_default_solver`), the tests, the forcing
calibration, the cost table, the 1024^3 Mach-5 reference
(`submit/examples/sil3d/ref.sbatch`, 1294 pooled snapshots) and the
barrier fix.

Paused (2026-09-28) in the second milestone, training: SiL training on
2 GPUs (mesh 2,1,1, 15-minute chunks) ran the K = 8 stage (200 iterations,
loss 7.1e-3 over the first 10, 5.7e-3 over the last 10) and 82 of 150
iterations at K = 32 (1.36e-2 to 1.11e-2); `run_m5_s7/sil/weights_K8.npz`.
The resumable training it used (Adam state in `state.npz`, `wall_min`,
`submit/examples/sil3d/chunk.sbatch`) is not on this branch.

Open: K = 32 and 128 to the end, the offline (one-step) baseline, the
evaluation beyond the training window (`sil3d.py eval`, `sil3d_figs.py`),
then the example's figures for docs/GALLERY.md.
