# Wall et al. 2020 M3 / M3f in LUNA+Torch

The validation pair against FLASH+Torch: Wall et al. 2020 (ApJ 904, 192;
arXiv:2003.09011), runs M3 (no feedback) and M3f (radiation + winds). The run
setup is `luna_m3.py` and `setup.py`, outside the repository.

| | Wall 2020 | M3 result | M3f result |
|---|---|---|---|
| stars at the end | | 514 Msun (51%), 1100 stars, 4.38 Myr | 338 Msun (34%), 1062 stars, 3.90 Myr |
| first star | | 2.86 Myr | 2.31 Myr |
| finest cell | | 0.01 pc | 0.02 pc |

LUNA runs both at 512^3 in the 10 pc box, dx = 0.0195 pc: M3f's resolution,
half of M3's.

## Parameters, and where each comes from

| quantity | value | source |
|---|---|---|
| M, R, D | 1e3 Msun, 3 pc, 10 pc | Table 1 |
| profile | rho_c exp(-(r/R)^2 ln 3), cut at R | Torch `utils/ic-generator/turb-sphere.py` (`rho_rat = 1/3`) |
| rho_c | 46.4 rho' = 1.109e-21 g/cm^3 (from M) | Table 1 says 46 rho': reproduced to 1% |
| turbulence | E(k) ~ k^-5/3 on k = 2..32 (k = 2 pi/D), cloud only | text + turb-sphere.py (`create_spspace_box`) |
| v0 | 0.616 km/s = mass-weighted 3-D rms in the cloud | gives Ekin/\|Epot\| = 0.205; Wall 2019 (arXiv:1901.01132, same runs): "virial ratio of kinetic to potential energy of 0.2" |
| mu | 1.3 both phases | turb-sphere.py defaults `--musph`, `--muamb` |
| cloud T | 46 K (uniform at t = 0) | Torch's equilibrium curve at the edge density (170 cm^-3) |
| ambient | 60 K, pressure equilibrium with the edge: n = 130 cm^-3, 3.7e3 Msun in the box | paper: "cold phase at roughly 60 K in pressure equilibrium"; the curve puts n = 130 at 57 K |
| sinks | r_acc = 2.5 dx, rho_sink = pi cs^2 / (G (5 dx)^2) at 46 K, softening r_acc | Wall 2019 ("2.5 times the smallest grid cell"), Torch docs/sinks, Polak 2024 eq. 2 |
| gas faces | diode (BOUNDARY 4) | see below: Torch's flash.par has FLASH "outflow" |
| gravity faces | isolated, multipole l <= 2 | Torch flash.par `grav_boundary_type = "isolated"` (FLASH `mpole_lmax = 10`) |

### "FWHM at R" is not what the code does

The paper says the Gaussian has its full width at half maximum at R. Torch's
generator instead sets rho(R) = rho(0)/3 and normalises the mass inside R.
Only the second reproduces Table 1: with M = 1e3 Msun inside R = 3 pc,
rho(R)/rho(0) = 1/3 gives rho_c = 46.4 rho' (Table 1: 46); half maximum at
R/2 would need 2.2x the central density and half maximum at R 0.81x. The
Lewis 2023 and Polak 2024 clouds use the same 1/3 ("central to edge density
3:1").

### v0

"v(k) = v0 k^-5/3" fixes no normalisation convention. Torch's generator
rescales the field to a virial ratio, and Wall 2019 states Ekin/|Epot| = 0.2
for these runs. Normalising the cloud's mass-weighted 3-D rms speed to v0 =
0.616 km/s gives Ekin/|Epot| = 0.205 (measured, 64^3 and 512^3), so that is
the reading used.

### The temperatures

Neither paper gives the cloud temperature. The one self-consistent point on
Torch's own heating/cooling equilibrium curve
(`hAc_b_2.0E-17_e_0.021_FUV_1.69.dat`, shipped with Torch, read as
`--Ts_from_cool_curve` reads it) is: the cloud edge (n = 170 cm^-3) at its
equilibrium 46 K, and an ambient in pressure equilibrium with it at the
paper's 60 K, which is n = 130 cm^-3, where the curve's own equilibrium is
57 K. The centre (n = 510) sits at 24 K on the curve; the mass-weighted
equilibrium temperature of the cloud is 33.5 K.

### Thermodynamics: Torch's equilibrium curve as the neutral EOS

LUNA has no low-temperature heating and cooling (its cooling table floors at
1e4 K), and the radiation runs prescribe the thermal state from the
ionisation fraction (`two_temperature`). With one neutral sound speed, a cloud
and an ambient cannot have different temperatures. `rad_neutral_eos_table`
makes the neutral sound speed a function of density, read from a table; the
M3 runs give it Torch's equilibrium curve (cut at 1e4 K: the neutral gas does
not go hotter, and held at the table's ends). On that curve the gas between
the ambient and the cloud centre has an effective polytropic index d ln P /
d ln n ~ 0.35: much softer than isothermal. What stays different from FLASH:
the equilibrium is instantaneous (no cooling time), and there is no FUV
shielding or dust temperature behind it.

## Boundaries

### Gas: diode, not FLASH "outflow"

Torch's `flash.par` sets `"outflow"` on all faces (with `"diode"` commented
out). FLASH's outflow is zero-gradient, which is LUNA's BOUNDARY 1: a face
where the interior gas moves inward feeds inflow at that speed. Polak 2024
say so, and chose it to avoid vacuum at the faces.

The M3 runs use the diode (BOUNDARY 4): the ghost's normal velocity is clamped
against inflow. That is not a zero-flux face: gas that gravity pulls away from
a face still opens a rarefaction that draws some mass in. Measured, 32^3 cloud
with isolated gravity, 6 steps: the box gains 6.7e-6 of its mass through the
diode, 1.35e-5 through the zero-gradient face (`tests/test_isolated_run.py`).
`res.json` `"boundary": 1` is the Torch-exact choice.

Over a run the difference is not small. Standalone LUNA (no Torch, no
feedback), the M3 cloud at 128^3, gas + sink mass in the box
(`wall_m3/bench/inflow.sh`, `mass_history.py`):

| t [Myr] | 1.0 | 2.0 | 3.0 | 4.0 | 5.0 |
|---|---|---|---|---|---|
| zero-gradient (1) | +12% | +55% | +157% | +432% | +2360% |
| diode (4) | +1.0% | +1.3% | +1.4% | +1.3% | +1.1% |

The whole box is self-gravitating (4.7e3 Msun, the ambient's own free-fall
time at 130 cm^-3 is ~4 Myr): pulled inward, a zero-gradient face feeds the
infall from an unlimited reservoir. With the diode the box gains 1.4% and then
nothing; in that run the three sinks hold 48% of the box's gas by 3.5 Myr and
98% by 5 Myr -- the ambient collapses too, which makes the ambient density
(not given by the paper) a first-order parameter of the comparison.

The 3-D diode now takes the wide hydro path (docs/HYDRO_KERNEL_PATHS.md): its
shard seams telescope (it did not before, audit 2026-09-26 M13) and the fused
Pallas divergence applies. Measured on one MI300A, FP64, minmod, HLLC:

| | 256^3 old | 256^3 new | 512^3 old | 512^3 new | weak, 2 GPUs x 256^3, per GPU old / new |
|---|---|---|---|---|---|
| BOUNDARY 4 | 747 | 2154 Mcell/s | 921 | 2175 | 500 / 2082 |
| BOUNDARY 0 | 2577 | 2595 | 2578 | 2591 | 2506 / 2498 |

The periodic path is unchanged (code and timing). The diode is 16% slower
than periodic at 512^3 (the clamp's two-layer writes).

### Gravity: isolated faces on one device and on a mesh

The multipole-Dirichlet solve (Guillet & Teyssier 2011 face-centred ghosts,
monopole + quadrupole) existed on one device only. Found and fixed on the way:

- **Face force (audit C6).** The gradient used the solver's zero-Dirichlet
  halo: 13x the true force at a 32^3 face (27x at 64^3). It now extrapolates
  the potential (quadratic for the 2nd-order gradient, cubic for the 4th):
  the centred difference becomes the one-sided one of the same order.
- **One FMG pass.** The FMG prolonged the coarse POTENTIAL with the
  corrections' zero ghosts. After the production cold solve (one pass, no
  extra V-cycles) the face force was 1.27x / 2.7x / 5.4x the true one at
  32^3 / 64^3 / 128^3 (worse with resolution). Prolonging by extrapolation:
  8.0e-3 / 2.7e-3 / 9.0e-4 (second order).
- **Warm V-cycles.** `gravity_warm_vcycles` (2) raised on code 3 (the fixed-cycle setting of that time).
  Now: the multipole halo at the top, Phi = 0 ON the faces (ghost = -edge)
  for the corrections. With the old zero ghost the warm cycle does not hold
  the solution (`test_warm_cycles_hold_and_reach_the_cold_solution` fails).
- **On a mesh** (`gravity/isolated.py`): the multipole moments are psum'd,
  the boundary shards write the face ghosts, the ladder is the one-device
  ladder level for level. Sharded = one device to 3e-15 (2x1x1 and 2x2x2),
  cold and warm; `ShardedGravity` and the coupled worker's
  `ShardedParticleBridge` both use it (bridge: 6e-15).
- **Depth on a mesh.** The default depth was the shard's, not the grid's: a
  4 x 32 x 32 slab of a 32^3 grid on (8,1,1) got ONE level, so every "V-cycle"
  was 50 Gauss-Seidel sweeps on the fine grid. Cold g was 87% off on the
  polytrope (one device: 0.4%), and the CPU-8 well-balanced polytrope drifted
  0.203 instead of 0.233. Now the depth is the grid's; the levels a shard
  cannot hold (sides under 4, or odd) run on the gathered grid, so every
  decomposition is the one-device solve to round-off
  (`tests/test_gravity_decomposition.py`). The periodic solve already
  gathered its coarse levels, but its cold FMG started from ONE V-cycle on the
  gathered grid (cold g 9% off at 32^3 on any mesh); it is now that grid's FMG.
- **Sharded gradient (audit M2)** wrapped periodically for every boundary;
  now edge (1, 2) or extrapolated (3), equal to one device (0 and 7e-15).
- **Sinks** read `boundary` as periodic-or-not plus a div-v halo that knows
  0/1/2: under code 3 they get 1 (ShardedSinks refused 3).

Accuracy against analytic answers (32^3 unit box): a uniform sphere, the
potential outside it to 5.1e-3 of GM/(L/2) and the face force to 5.7e-3; two
off-centre blobs against a direct sum, face potential to 6.1e-3 -- the l <= 2
truncation of the face values (6.7e-3 on the face plane itself, the
monopole alone 8.9e-2). FLASH uses l <= 10 there; raising LUNA's l_max is
open.

### Mass through the faces: the budget

`boundary_mass_budget=True` accumulates, per face (x_lo, x_hi, y_lo, y_hi,
z_lo, z_hi), the gas mass the hydro step moves in and out:
`state.sources["boundary_mass"]`, shape (2, 6), rows in / out, in the run
result as `extra["boundary_mass"]`. It is its own jitted shard_map program
(`integrators.create_boundary_mass_fn`): it recomputes the step's fluxes on
the six outer faces from four-cell slabs of the two-ghost pad
(`muscl_hancock.domain_face_fluxes_3d`), psums them over the mesh, and adds
them to the carry on the device. Off (the default) the step is untouched.
3-D, BOUNDARY 0/1/4; not checkpointed (a restart starts it at zero).

`tests/test_boundary_mass_budget.py`: the M3 cloud at 32^3 (v0 = 0.616 km/s,
isolated gravity, one sink formed at step 1), 20 CFL steps, 8 CPU devices
(2x2x2). (box gas + sink) - initial = in - out:

| | gain [Msun] | in | out | residual / M0 | each face (net in) |
|---|---|---|---|---|---|
| zero-gradient (1) | 374.6 | 374.6 | 0 | -3.3e-14 | 62.4-62.5 |
| diode (4) | 96.26 | 96.26 | 0 | -3.2e-14 | 16.0-16.1 |

Two processes of one device each (gloo, 2x1x1): -8.1e-14, equal to one
process of two devices. Teeth: dropping z_hi from the accumulator leaves a
residual of 62.4 (1) and 16.0 (4) Msun against a tolerance of 4.2e-9; the
test also checks that leaving out any one face breaks the budget. With the
accumulator on, the gas is bitwise the gas without it.

Cost when on, 512^3 FP64 diode on one MI300A: 4.12 ms per step next to the
step's 61.64 ms (6.7%; the extra two-ghost pad of the whole block
dominates).

After merging main (eb80392), `tools/bench_hydro_roofline.py`, 512^3, one
MI300A, ms per step, main -> this branch: periodic XLA 107.52 -> 107.87,
Pallas 52.10 -> 51.91; diode XLA 112.90 -> 117.76 (the wide path under
`hydro_kernel="xla"`, +4.3%), Pallas (the GPU default) 112.96 -> 61.61
(main's diode never reached the kernel).

On the diode every face lets mass in (as measured above: the rarefaction the
infall opens behind a face draws gas through it); nothing flows out in the
first 0.9 Myr (20 steps of ~0.047 pc/(km/s)).

## What remains different from Wall 2020

- no MHD (their 3 uG along z; deferred);
- thermodynamics: the equilibrium curve, not time-dependent heating/cooling;
- unigrid 512^3 for both (M3 had 0.01 pc AMR, M3f 0.02 pc);
- the same turbulence seed for m3 and m3f (Wall's two differ; theirs and ours
  are different realisations in any case);
- gas faces: diode (Torch: zero-gradient outflow; one switch);
- gravity face values l <= 2 (FLASH 10);
- sink creation: LUNA's density threshold + separation until the FLASH checks
  (converging flow, potential minimum, Jeans, bound, merging) land from
  branch sink-federrath; `luna_m3.py` has the place for them.

## Not pinned down by the papers

The cloud temperature and hence the ambient density (derived above from
Torch's curve, not stated); the v0 convention (derived from Wall 2019's virial
ratio); the random seeds; whether M3's IC used `--Ts_from_cool_curve`; the
IMF upper mass and the other Torch star-formation settings of 2020 (the runs
use the current Torch defaults from `flash.par`).

## Smoke test (2026-09-28, 512^3, 2 GPUs, apudev, tree e6a8730)

IC as the run builds it (`wall_m3/smoke_ic.py`): rho_c = 46.40 rho', n_c =
510, n_edge = 170, n_amb = 130 cm^-3; M(r <= R) = 999.99 Msun, box 4713 Msun;
cloud v_rms = 0.6160 km/s, bulk drift 0.005 km/s; Ekin/|Epot| = 0.205
(alpha_vir = 0.410), Eth/|Epot| = 0.473 at the uniform 46 K; t_ff = 2.72 Myr.
Shell means follow the Gaussian to 1% and the ambient exactly.

Coupled (Torch + PeTar + SeBa, m3f, `TEST=1`): initialize 73 s, the first
evolve 30 s (compile), then 0.12 s per bridge step (one hydro substep of
~2 kyr); 7.5 GiB in use per GPU, peak 18. 2.16 Myr in 11.5 minutes, no star
yet (Wall: first star at 2.31 Myr in M3f, 2.86 in M3).

Standalone LUNA on 2 processes (`python -m luna.main luna_m3.py` under srun)
stops after writing snapshot 0 and never reaches step 1 -- also periodic,
without radiation, without the EOS table (jobs 12016847, 12016948/49,
12017009/10). The coupled worker (same 2 ranks, same physics) steps; one
process standalone steps (the 128^3 runs above). Not investigated. Main's
sink-mp-hang (f2a5882, merged here) fixed a hang with that signature: the
progress line's sink report reduced on rank 0 only, and on the GPU the other
rank waited in compile. Not re-run on the M3 setup.

## Before the long runs: what changes when sink-federrath lands

Settled (2026-09-30): ambient n = 130 cm^-3 as in the paper and the diode
(BOUNDARY 4). Prerequisites: the sink Federrath checks. The setup is frozen on
`src_wall` = e6a8730 (the first wall-m3 commit). What has to change:

1. **The tree.** Refreeze `src_wall` from wall-m3-3 (wall-m3-2 merged with
   main a923490, which holds sink-federrath), update `src_wall/COMMIT`.
   `setup.py` TREE stays the path. The sink construction in
   `setup/assemble.py` carries both sink-federrath's three keywords and
   wall-m3's `boundary=1 if boundary == 3`; the hydro builder (and
   `build_face_mass`) now lives in `setup/hydro.py`.
2. **`SINK_FEDERRATH` in `luna_m3.py`.** Fill it with the names the branch
   has (not the placeholders there now): `sink_accretion="federrath"`,
   `sink_creation_checks="flash"`, `sink_merging=True` (Torch's flash.par:
   all checks and merging on), and `sink_star_T_max` if the Lewis+ 2023
   cold-gas condition (100 K) is wanted -- Wall 2020 predates it, so 0.
   `luna_m3.py` skips names the tree lacks with a print only: make a missing
   name an error, or a wrong tree runs LUNA's own sink rules silently.
3. **Ambient.** Set `amb_n_cm3: 130.0` in both `res.json` rather than
   relying on 0 = pressure equilibrium (which gives 130 today, from the edge
   at 46 K and the ambient at 60 K). `boundary: 4` is already there.
4. **Mass budget.** Turn on `boundary_mass_budget=True` in `luna_m3.py`: the
   runs gain mass through the diode faces (above), and the budget says how
   much and where. It is not checkpointed, so each chain link starts it at
   zero; log it per link. Note `floors: true` in `res.json`: the cosmo floor
   stack can add mass, which the budget will show as a residual.
5. **Smoke before submitting.** `TEST=1` for m3 and m3f on the new tree, and
   the standalone 2-process run above (the hang), before the 24 h chain.

## Launched (2026-09-30, tree 617362e on branch cloud-drift)

Done as listed above: `src_wall` refrozen from 617362e (main fcf7424 + the
cloud-drift IC fix + the worker's budget line; the old tree is kept as
`src_wall_e6a8730`); `SINK_FEDERRATH` holds the four real fields and a tree
without them raises; `amb_n_cm3: 130.0` in both `res.json`;
`boundary_mass_budget=True`. The worker prints one `LUNA mass:` line per
bridge step (gas, sinks, sinks gone out of the box, stars, winds, SN ejecta,
face in/out, the residual against the link's first step); with 2 GPUs the
worker's ranks print into `raw_<job>.txt`, and `chain.sh` appends each
link's last line to `budget.log`. `wall_m3/first_link.py RUN` summarises a
link (rates, first sink, the budget).

The cloud now has no bulk drift (`problem.zero_drift`, docs/TORCH.md); M3's
was 0.008 km/s, 2e-4 of E_kin, so the M3 cloud is otherwise unchanged
(v_rms 0.616 km/s, Ekin 189.7 before and after, 512^3).

Standalone, 2 processes x 1 GPU, `XLA_FLAGS=--xla_gpu_enable_command_buffer=`
`AMD_SERIALIZE_COPY=3`, apudev (job 12045960): steps past snapshot 0 (the
2026-09-28 hang is gone), 3840 steps to 2.57 Myr in 13 min, first sink
between 2.153 and 2.156 Myr, 3 sinks with 132 Msun at 2.57 Myr.

m3, first link (12045987, apu, 2 GPUs), measured: 128 ms per LUNA step
before sinks; 4.2-5.7 Myr per wall hour, of which ~60 s per 0.1 Myr is the
checkpoint write; first sink at 2.154 Myr (the standalone run's time); mass
budget residual |resid| <= 1.8e-12 Msun (4e-16 of the box) through the
first sinks; 15.1 Msun in through the diode faces by 2.17 Myr, 2.37 net per
face, 0.87 out.

`ff7ec03` (the stricter bound check) changed only the `luna` creation mode;
the `flash` mode M3 uses has had the same check since sink-federrath
(a38b3d6), so it does not move M3's first sink. On the Burgers study cloud
(`luna` mode, 256^3, drift-free IC, apudev) it did: first sink at 0.690 Myr
with it, 0.580 Myr with ff7ec03 reverted (0.11 Myr later).
