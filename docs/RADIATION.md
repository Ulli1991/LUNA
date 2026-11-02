# Radiation in LUNA

Grey or multigroup M1 transport (`luna/radiation/m1.py`), coupled to the gas
as an operator-split stage (`luna/sources/radiation.py`), with hydrogen
ionization as the STARBENCH tests define it (`luna/radiation/ionization.py`).
What differentiates and what a gradient costs is in `docs/AUTODIFF.md`.
M1 is the default; `rad_closure="m2_learned"` swaps in ten-moment M2 with a
learned closure, where beams cross (last sections: "M2 as a stage").

## The solver

The state is `(G, 1+ndim, N...)`: G frequency groups, each carrying an
energy density and a flux. The group axis is optional -- a grey state
`(1+ndim, N...)` is one group and comes back in the same shape. Levermore's
M1 closure, a GLF face flux, minmod-MUSCL reconstruction and SSP-RK2, at a
reduced speed of light that the caller picks.

The closure is evaluated as chi = (5 - 2 sqrt(4 - 3 f^2)) / 3, which is
Levermore's (3 + 4 f^2) / (5 + 2 sqrt(4 - 3 f^2)) without the division, and
one reciprocal 1 / (c_hat E |F|) gives both f and the direction 1/|F|
(`m1._closure`, shared by the XLA and the Pallas sweep). Since 2026-10-11;
before that it was three divisions per face state. A round-off-level change,
not a bitwise one: one A100 split step at 256^3 moved by at most 1.1e-15
relative (flux) and 3.1e-16 (energy); the gate numbers (STARBENCH Stromgren,
diffusion rates, diagonal beam, beam crossing) are unchanged to the digits
the tests print. The fused sweep is 1.23x faster for it (A100, FP64).

Two properties the scheme needs and now has explicitly:

- **`|F| <= c E`.** The Eddington factor is a function of f = |F|/(cE) on
  [0, 1]; f > 1 is not a state the model has, and a transport step can leave
  one. `m1_floor` rescales the flux back onto the cone at the end of every
  step (`rk2`) or every sweep (`hancock_split`). Without it a point source at 144^3 reached a photon density of
  1e200 within a few hundred substeps while the same run at 96^3 was stable.
- **The square root of the flux magnitude is softened**, because
  d|F|/dF is infinite at F = 0 and reverse mode turns that into NaN.

Multi-GPU is the hydro's own ppermute halo injected into the transport (the
`pad` argument), not a second implementation; the two agree to 1e-12.

With `rad_kernel="pallas"` (or "auto" on a GPU; the default is "xla"),
`rad_scheme="hancock_split"` runs each grey 3-D sweep (slopes, closure,
Hancock predictor, GLF flux, update, floor) as one fused Pallas/Triton kernel
(`luna/radiation/m1_pallas.py`), bitwise equal to the XLA sweep (on NVIDIA one cell per
thread, tile 1x4x32 with 4 warps: 1.4x the radiation step on an A100, bitwise). The floor
runs after every split sweep, not once per step, because the Hancock
predictor overshoots to E < 0 at a sharp front and that compounds across
sweeps; since 2026-10-11 there is no second floor after the last sweep (XLA and Pallas
alike), which only ever moved cells with c_hat E below E_FLOOR (round-off; CODE_NOTES.md).

On an axis one device holds whole (all of them on one device; the unsharded ones of a
mesh, `pad.local_axes`) the fused split step (`m1_pallas.split_step`) does without the XLA
pad before each sweep: the x and y sweeps read the unpadded state and clamp (edge) or wrap
(periodic) the plane index along the sweep in the kernel, which needs a tile of one plane
along it (4x1x32 for y on NVIDIA); the sweep before a z sweep writes its output with two
z ghosts each side, which a small kernel fills in place. Up to
`m1_pallas.PREDICT_ONCE_MAX_CELLS` (2^25 cells per device) a sweep is two kernels: the
Hancock-predicted edge states of every cell once, then the faces, update and floor; above
it, the one fused kernel, which predicts each cell three times but keeps no (8, N..)
buffer (+1.6 GiB at 256^3, ~+13 GiB at 512^3). Sharded axes take the halo exchange as
before. All of it is bitwise the padded fused sweeps.

## Coupling to the gas

`luna/radiation/coupling.py`, exactly conservative cell by cell so the gates
assert on totals rather than tolerances:

- the transport substeps hand back the flux the opacity destroyed;
- the force gives the gas that momentum, with the true speed of light and
  not c_hat (the RSLA slows transport, not coupling), adds the kinetic
  energy it creates to E_tot so the internal energy is untouched, and takes
  that work out of the radiation energy;
- absorption and emission move energy between the groups and the gas through
  the exact exponential, stable at any optical depth.

## The stage

`luna/sources/radiation.py` builds the field and the two half-steps of the
`Radiation` stage. A half-step is n_sub x (transport, flux damping), then the
force on the gas, then absorption and emission; it runs in that order before
the hydro and in reverse after it, so the step stays a palindrome.

- **Step bound.** The transport is explicit at c_hat. The substep cap n_sub
  is a static int (a per-step count would retrace), so the stage bounds the
  hydro step at dt <= 2 n_sub per_sub, with per_sub = cfl dx/(ndim c_hat) for
  `rk2` and 2 cfl dx/c_hat for `hancock_split`. A shorter step uses
  ceil(h/per_sub) substeps; under a trace it uses the cap.
- **One program per substep.** A half step runs as n calls of the
  one-substep program over h/n: a loop inside one program needs more scratch,
  and async dispatch holds every queued program's scratch at once. Up to
  `LAG_MAX_CELLS` (2^25 cells per device) the next substep is queued while the
  previous one runs (a wait on the previous substep's inputs, so at most two
  programs' scratch); above it each call waits for its inputs. With ionization the
  substep program advances only (R, x), reading rho from U and taking the source
  weights as an argument, and the two-temperature pressure is set once per half
  step: it reads rho, the momenta and the final x, and the transport reads only rho.
- **Dark skip (`SKIP_DARK`).** While nothing shines, no light is left and
  x = 0, a half step maps zeros to zeros; it is skipped and only the
  prescribed two-temperature pressure is applied.
- **Ionized CFL.** With ionization on, the stage also bounds dt at the hydro
  CFL with the ionized sound speed as soon as anything shines: the hydro's
  own CFL sees the pressure before ionization, which raises it about 10x.
- **Multi-GPU.** The whole half-step is one shard_map using the hydro's
  ppermute halo; on one device it is a plain jit.

## Ionization, and the STARBENCH test

`rad_ionization=True` switches the stage to the benchmark's prescription
(Bisbas et al. 2015, MNRAS 453, 1324): pure hydrogen, case-B recombination
on the spot, and a temperature that follows the ionization fraction instead
of an energy equation -- deliberately, so that codes are compared on their
hydrodynamics rather than their chemistry. The photon field is the M1 state
read as a photon NUMBER density, which makes absorption exact in the only
currency that matters: one absorbed photon is one ionization.

Absorption attenuates E and F by the same factor and runs at every transport
substep. Giving the absorption opacity to the flux damping instead -- the
scattering operator -- destroys the flux of a cell that is optically thick
to the step while the capped absorption leaves its photons in place: the
radiation then stops streaming and piles up (measured: 59% of every photon
ever injected still in the box, and a Strömgren sphere 15% too small).

**The Strömgren radius**, with the source alone and the gas held fixed,
converges to the analytic 0.314 pc: +5.6% at ten cells per R_St, +2.4% at
twenty, +1.6% at thirty. That is `tests/test_radiation_ionization.py`.

**The D-type expansion**, the benchmark's early-phase test: 1e49 photons/s
switched on at t = 0 in a uniform 5.21e-21 g/cm^3 medium, ionized gas at
1e4 K, neutral at 1e2 K, run to 0.141 Myr.

    python -m scripts.validation.runs.starbench_hii_run nx=96 box_pc=3.0
    python -m scripts.validation.plots.starbench_hii_plot snapshots/starbench_early_nx96

96^3 in a 3 pc box takes 84 s on one MI300A; 240^3 in a 5 pc box takes
about half an hour. The front tracks Hosokawa-Inutsuka closely and ends
above it, and the answer converges:

| run | R(t_end) | R / Spitzer | R / Hosokawa-Inutsuka |
|---|---|---|---|
| 96^3, 3 pc | 1.372 pc | 1.091 | 1.012 |
| 144^3, 3 pc | 1.443 pc | 1.148 | 1.065 |
| 240^3, 5 pc | 1.454 pc | 1.157 | 1.073 |

The last two agree to 0.8%, and the bigger box barely moved the answer, so
what the first run was short of was resolution and not room: the shell is
the hard part, its density jump being (c_i/c_o)^2 ~ 200, and STARBENCH says
the same about the spread between its own codes. The reduced speed of light
is not the explanation either -- raising it from 30 c_i to 100 c_i moves the
front by 0.7%.

**So the converged answer is about 7% above Hosokawa-Inutsuka**, which puts
it just outside the bracket the twelve STARBENCH codes sit in rather than
inside it. That is the honest state of this test. The candidates, none of
them yet measured against each other: the thermal prescription below, which
warms partially ionized cells; the photon-conserving absorption at the
front, which puts the ionization one cell ahead of where a sharp-front
solver would; and the source region, a sphere of two cells rather than a
point.

A third, smaller one: the prescribed sound speed is interpolated linearly in
the ionization fraction, cs^2(x) = cs_o^2 + (cs_i^2 - cs_o^2) x, which makes
a partially ionized shell cell slightly warmer than a sharp two-temperature
rule would.

`rad_ion_solver="implicit"` (the default) solves absorption and
recombination together by backward Euler. With a = c sigma n_H dt and
b = alpha_B n_H dt, the neutral fraction z = 1 - x satisfies
(1 - z) = x0 + (N0/n_H) a z/(1 + a z) - b (1 - z)^2, monotone with one root
in [0, 1], solved by Newton inside a bracket bisected in log z (g and g' share
one reciprocal 1/(1 + a z) since 2026-10-11, a round-off-level change, 1.2x on an
A100; CODE_NOTES.md). It is solved in z, not x, because a reaches 1e9 in dense cells and the root can sit at
z ~ 1e-15; solving in x, or bisecting linearly, let photons leak through
cells and the front ran away. `"split"` (`ionize`, subcycled
`rad_ion_subcycles` times) does absorption then recombination, which is not
converged when the recombination time is shorter than the step.
`rad_neutral_eos_table` names a two-column (rho, cs^2) file that makes the
neutral sound speed density dependent. Many point sources are CIC-deposited
(`deposit_sources`) with the same clamp/wrap rule as the gravity deposit.

## M2: the maximum-entropy closure, differentiated implicitly

`luna/radiation/m2.py`, slab geometry. M1 carries (m0, m1) and closes m2
with a formula; it cannot tell a beam from the two opposite beams that make
the same m1, so it gives them the pressure of an isotropic field and they
collide. M2 carries m2 and closes m3 by maximum entropy: psi(mu) =
exp(a0 + a1 mu + a2 mu^2), with the multipliers found per cell from the
moments. By homogeneity the problem is two-dimensional in f = m1/m0 and
g = m2/m0, and it is convex -- the potential log m0 - a f - b g has the
residual as its gradient and a symmetric positive definite Hessian -- so a
damped Newton from the isotropic start converges.

**The closure is differentiated by the implicit function theorem, not by
walking the Newton loop.** The rule is given as a JVP, which is linear in
the tangent, so JAX transposes it to get reverse mode: both directions come
from the one argument, where a custom VJP would have made forward mode an
error. Measured on a four-step rollout at 128 cells, gradient seconds
against the Newton iteration count:

| iterations | implicit | unrolled |
|---|---|---|
| 10 | 0.120 | 0.073 |
| 30 | 0.117 | 0.224 |
| 100 | 0.114 | SIGSEGV |

The implicit gradient is flat in the iteration count and the unrolled one is
linear in it, crossing over between 10 and 30 iterations and, at 100,
producing a program large enough to hit the ROCm dispatch crash documented
in `docs/AUTODIFF.md`. The two agree to every printed digit wherever both
run. Peak memory does not separate them at this size: the process
high-water mark is 144 MiB either way, dominated by the runtime.

**Two beams is the test that separates the closures** and it is in
`tests/test_radiation_m2.py`: a packet with m1 = 0 and m2 = m0. M2 splits it
into two beams that travel at +-ct, giving a spread of sqrt((ct)^2 +
sigma^2) -- measured 0.158 against 0.158 predicted -- while M1 sees only
m1 = 0, calls it isotropic, and spreads it at c/sqrt(3) without moving the
peak: 0.106 against 0.102 predicted. Nine gates in all, including the
entropy problem's own residual (1e-11), the known limits, the moment bounds,
the implicit gradient against finite differences and against unrolled AD,
the transpose identity, and a differentiable rollout.

## The learned closure

`luna/radiation/m2_learned.py`. The expensive part of M2 is the per-cell
entropy solve, and the tempting fix -- fitting m3 as a function of (f, g)
-- has no reason to correspond to any distribution, so it can leave the
moment bounds and cost the system its hyperbolicity. What is learned here
is the convex dual instead: an input-convex network h_theta (non-negative
weights on the hidden-to-output path, softplus activations) whose GRADIENT
is the multiplier map, with m3 then read off the distribution exp(a.v).
|m3| <= m2 then holds whatever the weights are -- an untrained network
still gives a legal closure, which `tests/test_radiation_m2_learned.py`
checks along with the convexity and the hand-written input gradient.

That gradient is written out rather than taken with `jax.grad` for a
reason: training it would otherwise be second-order autodiff, and on this
ROCm stack the gradient of a function containing a gradient segfaults --
reproducibly, for a 24x24 network, where the CPU backend runs the same
program. With the Jacobian propagated forward by hand everything stays
first-order and, as it turns out, the gradient with respect to the
PARAMETERS still crashes on ROCm while the value does not. Training
therefore runs on the CPU backend, which for 1.3k parameters is seconds.

Measured (CPU backend, 12000 Adam steps on 1024 sampled states, then 200
steps of solver-in-the-loop fine-tuning against the exact-closure rollout):

| | closure error, max / rms | rollout error at 40 steps, max / rms |
|---|---|---|
| after pointwise training | 1.9e-2 / 3.1e-3 | 2.6e-2 / 1.2e-2 |
| after in-the-loop training | 1.7e-2 / 5.4e-3 | 1.6e-2 / 8.9e-3 |

Training through the solver lowers the trajectory error by 26% while
leaving the pointwise error where it was, which is the point of doing it:
the closure is being asked for the trajectory it produces rather than for a
pointwise fit. Over the 12-step window it was trained on, the relative
rollout mean square error fell from 9.1e-7 to 2.1e-7.

**What it costs.** Two backends, and they do not agree, so both are
given. Per step, 1024 cells, everything in each row measured in the same
run:

| | CPU | x M1 | GPU | x M1 |
|---|---|---|---|---|
| M1 (closed-form entropy closure) | 0.056 ms | 1.0 | 0.139 ms | 1.0 |
| M2, entropy solve per cell | 367.6 ms | 6551 | crashes | -- |
| M2, network for the multipliers | 6.63 ms | 118 | 0.256 ms | 1.84 |

On one backend, CPU, where both the solve and the network run: **the
network replaces the per-cell entropy solve at 55x less cost** -- 6551x
M1 down to 118x M1 -- at a closure error of 3.1e-3. An earlier version of
this section put a GPU network time against a CPU solve time and claimed
three orders of magnitude; that comparison was invalid.

The network is also far more backend-sensitive than M1: 1.84x M1 on the
GPU against 118x on the CPU for the same computation, because M1 is cheap
on both while the network's matmuls and quadrature only pay off on a GPU.
And it must be called on the whole field at once -- mapping it over cells
with `jax.vmap` costs 30-50x more on the GPU for identical arithmetic,
which is a trap this repository fell into once already.

## Why this is not yet a recommendation to switch

(Written for the slab code; the three-dimensional answer is "M2 as a stage"
below.) Everything above is SLAB geometry. Three dimensions is where it would
matter, and it is not done, for two reasons worth keeping in view.

**The field count.** M1 carries (E, F_x, F_y, F_z) = 4 fields and the M2
system carries (E, F_i, P_ij) = 10. The same unsplit stencil with no
closure in either, one MI300A, FP64: 0.156 ms against 0.266 at 64^3,
0.676 against 1.599 at 128^3, 1.831 against 4.492 at 192^3 -- a floor of
about 2.4x before any closure is evaluated. It is bandwidth rather than
arithmetic, so it multiplies by the number of frequency groups, in memory
and in every halo exchange.

**The entropy problem grows.** In slab the closure solves for two
coefficients; in three dimensions it is eight, against a spherical
quadrature, in every cell. That is what makes the M2 closure expensive,
and it is the whole reason to want a network standing in for the solve.
Both now exist -- `m2_3d` and `m2_learned_3d` -- and the section below
says what they measure.

**And the closure is not the limiting error in the problem this code
actually ran.** The converged STARBENCH D-type expansion above is 7% from
the reference, and none of that 7% is the closure: it is the shell
resolution, the thermal prescription and the treatment of the front.
M1's real failure -- beams merging -- bites in optically thin regions
with several discrete sources, which production codes route around with a
radiation field per source or by ray tracing the point sources.


## Three dimensions, measured

`m2_3d` is the entropy closure on the sphere: eight multipliers found by
damped Newton against a 1152-point Gauss-Legendre rule, per cell.
`m2_learned_3d` replaces that solve with an 8-64-64-7 MLP predicting the
seven free components of the trace-free part of m3, with the trace part
imposed algebraically so `m3_iij = m1_j` holds exactly whatever the
weights are. Unlike the slab network it predicts m3 directly, not the
gradient of a convex dual, so no hyperbolicity argument comes from the
architecture, and it is rotation-equivariant only as far as training on
randomly oriented states taught it. Its output is squashed into
(-BOUND, BOUND): an unbounded head once handed the transport a superluminal
flux. Inference is FP32; the closure error is about 1e-2, so FP64 buys
nothing.

The check worth trusting is that the 3D solver and the slab solver, which
share no code, agree to 1e-10 on states symmetric about an axis.

**Beam crossing, one MI300A, FP64, two Gaussian beams meeting head-on,
run to ct = 0.2 against the analytic answer (the beams ignore each
other):**

| | M1 err | M1 peak | learned M2 err | learned M2 peak | exact peak |
|---|---|---|---|---|---|
| 64^3 | 0.595 | 3.114 | 0.056 | 1.809 | 1.987 |
| 128^3 | 0.682 | 3.530 | 0.048 | 1.888 | 1.997 |

M1 does what it must: with m1 = 0 in the overlap it cannot tell two beams
from isotropic radiation, so it applies isotropic pressure and the energy
piles up -- and the error GROWS with resolution, 0.595 to 0.682, because
the pile-up sharpens. The learned M2 passes the beams through and its
error falls with resolution.

**Cost, 64^3, each layer added on top of the last:**

| | ms/step | x M1 |
|---|---|---|
| M1, four moments | 0.714 | 1.00 |
| M2 ten-moment transport, algebraic closure | 1.650 | 2.31 |
| + realizability projection | 4.646 | 6.51 |
| + network, FP32 | 14.576 | 20.42 |
| + network, FP64 | 16.535 | 23.16 |

Only the last two rows are the closure. 2.3x is carrying ten moments
instead of four and would be there with any closure at all. FP32 for the
network buys 12%, which says the closure is bandwidth-bound -- it writes
a 27-component tensor per cell per face -- not arithmetic-bound.

The entropy closure is not in that table because it is four orders of
magnitude away. Per step, one MI300A, FP64, all three built and timed in
one process:

| | M1 | M2 entropy | M2 learned | learned x M1 |
|---|---|---|---|---|
| 32^3, serialized | 0.217 | 28383.9 | 2.841 | 13.1 |
| 64^3, serialized | 0.754 | 218760.4 | 17.153 | 22.7 |
| 32^3, free-running | 0.212 | crashes | 3.027 | 14.3 |
| 64^3, free-running | 0.752 | crashes | 15.041 | 20.0 |
| 256^3, free-running | 46.344 | ~3.9 h (scaled) | 631.397 | 13.6 |

Serialized launches are the only mode the entropy closure survives, so
those are the rows where all three numbers were taken under the same
conditions and the ratios mean something; the free-running rows are what
M1 and the network actually cost in production. The 256^3 entropy entry
is scaled from the 64^3 measurement, not measured -- it is 3.9 hours per
step, which is why it is not there.

**The learned closure against the entropy closure it approximates: 9990x
at 32^3 and 12754x at 64^3**, and the ratio improves with resolution
because the entropy solve is a fixed cost per cell while the network
amortises. That is what the module exists for.

Against M1 it is 13-23x, and that factor is mostly not the closure. A
single closure evaluation at 64^3 -- 262144 cells -- decomposes as:

| | ms per evaluation | x12 = per step |
|---|---|---|
| the MLP alone, FP32 | 0.388 | 4.66 |
| + building the 27-component tensor | 0.497 | 5.96 |
| + targets, reshapes, times m0 | 0.521 | 6.26 |

Getting the seven coefficients costs 0.39 ms for a quarter of a million
cells. The step asks for them TWELVE times -- two RK stages, three axes,
two face states -- because the code closes on the reconstructed face
states rather than at cell centres. Closing at cell centres and
reconstructing m3 to the faces would be 2 evaluations, and the
realizability projection's unconditional eigendecomposition is another
3.0 ms that has nothing to do with the closure at all. Neither change is
made here.

### The realizability bug this found

`realizable()` normalised the trace of the second moment and not its
spectrum. T = m2/m0 is an average of n(x)n, so its eigenvalues are in
[0, 1] and sum to 1 -- but a trace-only projection lets one eigenvalue go
negative while another compensates, which keeps the trace at exactly 1
and leaves nothing to push it back. On beam crossing at 32^3 that reached
eigenvalues of -71 by step 44 and overflowed by step 48, with m0 and
|m1|/m0 perfectly well behaved the whole way. It looked like the network
being unstable. It was not.

The projection now clips the spectrum. It is applied as a stopped
correction, `T + stop_gradient(P - T)`: the projection is the identity on
the set the physics lives in, so that is its derivative there, and
differentiating through `eigh` would return NaN wherever eigenvalues are
degenerate -- which is every isotropic cell.

### What does not run

At 256^3 the learned closure segfaults on this ROCm 7.2 stack for every
block size but the largest, which merely runs too slowly to finish. M1
runs at 256^3 fine.

The fault is specific and reproducible: more than two evaluations of the
closure in one compiled executable segfault at execution. One axis (two
face states) survives at 64^3; two axes crash. It is not the size of the
data -- two evaluations run fine at 128^3 -- not fusion, since an
`optimization_barrier` between them changes nothing, not the allocator,
and not asynchronous dispatch, since `HIP_LAUNCH_BLOCKING` does not help
and compilation succeeds before the crash. The workarounds are in the
code: `third_moment(..., block=)` runs the closure through a `lax.map`
over cell blocks, and `make_split_step` compiles one axis at a time.
Neither is wanted for its own sake; `block=0` and `step` are the readable
versions and are what the tests use.

## M2 as a stage: `rad_closure`

    rad_closure = "m1"            # the default: Levermore M1, 4 moments
    rad_closure = "m2_learned"    # 10 moments, the learned closure: beams cross
    rad_closure = "m2_entropy"    # 10 moments, the exact closure: a reference only

`python main.py myrun.py rad_closure=m2_learned`, or the field in the
config. It needs `SPATIAL_DIMENSION=3` and `rad_scheme="hancock_split"`
(the default), and there is no adjoint path yet; `validate()` refuses the
rest. `rad_closure_params` names another weights file; empty is the shipped
8-32-32-7 network (`luna/radiation/m2_closure_w32.npz`, 14 kB), trained by
`scripts/validation/runs/m2_learned_3d_train.py` (12000 entropy-closure
states, held-out m3 error rms 1.0e-2). `m2_entropy` solves the entropy
problem at every face state -- about 4 h per step at 256^3 (scaled from the
64^3 measurement above), and on ROCm it needs serialized launches -- so it
is there to check the learned one against.

**What the stage carries.** A group is (m0, m1_i, m2_ij), ten fields where
M1 has four, with m0 = E and c_hat m1 = F, so the force on the gas, the
energy bookkeeping, the ionisation and the checkpoint read an M2 state as
they read an M1 one. The transport is M1's split structure on ten moments
(`m2_split`: x y z z y x Hancock sweeps, floored after each). Two coupling
terms have to say what they do to m2: the opacity damps m1 and the
trace-free part of m2 together (isotropic scattering and absorption both
remove the anisotropy of the intensity at the same rate), and energy moved
in or out of m0 -- absorption and emission, the work of the force, photons
injected by a source -- is isotropic, a third on each m2_ii, so trace(m2) =
m0 stays exact. Photoionisation scales every moment by the same factor, as
it scales E and F for M1. Conservation is the same statement as for M1 and
is gated the same way: gas plus radiation energy and momentum to round-off
through both halves of the stage, and through real steps of the run.

**The closure is evaluated once per transport step.** The network gives the
seven free components of the trace-free m3 over m0 at each cell centre at
the start of the step; they are held over its six sweeps, constant over
each cell, and every edge state (reconstructed or predicted) gets
m3 = trace part of its own m1 + that normalised trace-free part times its
own m0. Measured on the crossing below (32^3, 64-64 network, the XLA sweep):
closure at every face state 0.0915, at the cell centres twice per sweep
0.0914, once per step 0.0899 -- evaluating it once is not a loss. Holding it
constant over the cell rather than reconstructing it with minmod: 0.0918
against 0.0939 (32-32 network), and 6% cheaper at 512^3; that is what is
kept.

**Beam crossing** (`tests/test_radiation_m2_stage.py`, through the stage):
two Gaussian beams head-on along x, 32^3, 11 split steps to ct = 0.2,
relative L2 error against the analytic answer (the beams ignore each
other):

| | error | mid-plane peak (exact 1.950) |
|---|---|---|
| M1 | 0.4398 | 2.372 |
| M2, trace closure (network off) | 0.2073 | -- |
| M2 learned, 32-32 (shipped) | 0.0918 | 1.607 |
| M2 learned, 64-64 | 0.0870 | 1.668 |
| M2 entropy (earlier run, every face state) | 0.0839 | 1.649 |

The test asserts M1 > 0.40 (the pile-up it is built on), learned M2 < 0.11
and below a quarter of M1. The trace closure and a closure row taken along
the wrong axis both fail it (0.207, 0.223). The 64-wide network is 5%
better and its network costs 3.2x as much (below), so the 32-wide one
ships. M1 and the shipped network give the same numbers on the CPU backend
(the XLA sweep) and on the MI300A (the fused kernels), to the digits shown.

**Cost, 512^3, one MI300A, FP64 moments, ms per split step** (median of 10,
`tools/bench_m2_split.py`; both schemes bound by the same CFL, so this is
cost per unit simulated time):

| | M1 | M2, trace closure | M2 learned | learned x M1 |
|---|---|---|---|---|
| 512^3, fused kernels | 71.05 | 112.35 (1.58x) | **148.75** | **2.09** |
| 512^3, XLA | 316.68 | out of memory | out of memory | -- |
| 256^3, fused kernels | 8.61 | 14.52 (1.69x) | 19.08 | 2.22 |
| 256^3, XLA | 40.81 | 86.52 (2.12x) | 108.05 | 2.65 |
| 128^3, fused kernels | 1.17 | 1.86 (1.60x) | 2.54 | 2.18 |
| 128^3, XLA | 5.05 | 11.44 (2.27x) | 14.05 | 2.78 |

The XLA M2 sweep does not fit on one MI300A at 512^3: it asks for 70 GiB
in one allocation (it materialises every intermediate of ten moments, as
the XLA M1 sweep does for four). It is the fallback (CPU, tests); the
default on a GPU is the fused one.

**Where the 2.09x goes, 512^3:** the ten-moment transport with the trace
closure is 1.58x M1; the network, once per step, 16.72 ms = 0.24x M1 (the
kernel; XLA's is 40.56 ms); the rest, 0.27x (the difference), is the
frozen closure's seven fields read by every sweep and padded once per step. Two variants that were
measured and not kept (512^3, fused sweep, one probe run in which M1 took
68.01 ms): the network in XLA, 175.9 ms (2.59x, with the constant closure);
the closure reconstructed with minmod, 155.0 ms (2.28x, with the network
kernel). A 1x8x64 tile was 22-27% slower than 1x4x64 at 512^3; 1x2x64 and
1x4x32 were 30-35% slower at 256^3. The 64-wide network's kernel costs
53.14 ms at 512^3 against the 32-wide one's 16.74.

XLA's network materialises both hidden layers (16 GiB each at 512^3 at
once, so it runs on slabs of 2^24 cells); the kernel keeps them in
registers with the weights compiled in as constants. The earlier kernels
that lost to XLA read the weights from memory at every multiply-add (the per-face in-kernel network: 188x M1 at
64^3) or ran the hidden layer through the matrix cores (61 ms against XLA's
4.6 at 64^3); both are deleted. Against the network in FP64 (numpy, 2^20
cells of the bench state, outputs up to 0.08): the kernel 2.2e-7, XLA's
default FP32 matmul on the MI300A 5.9e-4. `precision="highest"` is worse
on this stack (1.8e-2, measured on held-out states), and XLA's FP64 matmul
fails to autotune on ROCm. So on a GPU the fused step and the XLA step
differ by that network error (2.1e-5 of the state after one step at 8^3; on a CPU,
below 1e-6).

**A100, 2026-10-11** (FP64; CODE_NOTES.md). The fused M2 split step does without the
XLA pad as M1's does (`m2_pallas.split_step`: x and y sweeps clamp or wrap the plane index
in the kernel, z ghosts written by the sweep before and filled in place, sharded axes take
the halo) and on NVIDIA runs one cell per thread (tile 1x1x128, 4 warps; MI300A keeps
1x4x64): at 256^3 the trace step 37.4 -> 24.3 ms and the learned one 56.4 -> 42.7 ms,
bitwise, at one more state of scratch (temp 1.27 -> 2.54 GiB: the two z sweeps hold two
z-padded states). The XLA sweeps do without the predictor-flux barrier (bitwise, same memory):
37.5 -> 35.2 ms (trace) and 43.4 -> 41.7 (learned) at 128^3. The floor takes one reciprocal of m0 for its seven
divisions by it (round-off: one step moves by at most 1.8e-15 of max|U| per component; the
crossing stays 0.0919): fused 24.2 -> 22.9 ms (trace) and 42.6 -> 41.5 (learned) at 256^3,
XLA 35.2 -> 28.9 and 41.7 -> 35.2 ms at 128^3.

**What is not done.** No adjoint (the fused kernels have no derivative and
the floor's arccos has not been checked under AD); one network for every
group of a multigroup run; the ionisation kernel (`m1_pallas.ionize`) is
M1's four-moment one, so an ionising M2 run uses the XLA absorption.

## The thermal state is prescribed, not computed

Worth being explicit about, because the ionization path looks more complete
than it is: **nothing in the radiation module heats the gas.**
`two_temperature` overwrites the pressure with rho*cs^2(x), interpolating
between a neutral and an ionized sound speed by ionization fraction. There is
no photoheating rate anywhere -- no excess energy per ionization, no
temperature that comes out of a balance.

That is STARBENCH's own prescription, and it is the right one there: the
benchmark isolates the hydrodynamics of D-type expansion from the
thermodynamics, so every code compared is solving the same problem. The H II
region result above is therefore a statement about the dynamics and says
nothing about the thermal structure.

For a molecular cloud it is a placeholder, and a dangerous one, because it
writes the same variable the cooling stage does and neither knows about the
other: whichever ran last wins, per cell, and the answer looks plausible.
`Config.validate` now refuses `rad_ionization` together with
`cooling_enabled` for that reason.

Lifting it, in the order that actually blocks physics:

1. A real photoheating rate -- the excess energy per ionization deposited as
   heat -- so the radiation heats rather than prescribes. This is the one
   that makes `two_temperature` retirable and the guard removable.
2. Reconciling that with the cooling stage, so the two compose instead of
   overwriting.
3. An FUV band and a G0-dependent photoelectric heating rate. LUNA has the
   band structurally (multigroup M1) and has a PE term already, but
   `sources/cooling.py` carries it as a SCALAR (`heating: float`, erg/s per H
   atom, Koyama-Inutsuka), not a field. Making it a function of the local FUV
   energy density is a scalar becoming a field in a term that already exists.
4. Dust, then molecular cooling, before the cold phase means anything.

## Restart

The radiation carry is written beside the grid checkpoint of the same step,
as `radiation_{step:06d}.{pid}.hdf5` in the same shard format
(`luna/file_io/radiation_checkpoint.py`). It is flattened to U's rank and
sharding first: `R (groups, m, Nx, Ny, Nz)` becomes `groups*m` planes (m = 4
for M1, 10 for M2), then one plane each for `x` and `w` when ionization is
on. `rad_groups`, `rad_moments` and `ionization` are stored as attributes,
so unpacking never reads the config and a changed `rad_groups` cannot
silently reinterpret the file. Files written before M2 have no `rad_moments`
and are read as M1.
