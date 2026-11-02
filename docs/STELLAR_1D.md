# Stellar 1D hydro

The standalone FP64 CGS path now has a runner: `python main.py stellar --help`.
It supports planar and spherical Helmholtz hydro, aprox13 burning, spherical Newtonian
self-gravity, HDF5 checkpoints, and restart. It does not use the production gamma-law
assembly, AMR, Pallas, or mesh decomposition. Those paths are unchanged.

## Method and diagnostics

`hydro.stellar_1d.make_solver` evolves `(16, cells)`: density, momentum, thirteen species
mass densities, and total gas energy last. Temperature, pressure and acoustic speed come
from the Helmholtz EOS. Its temperature inversion uses an implicit first derivative.
Full reactive-solver sensitivities remain unvalidated.

The default is conservative MUSCL–Hancock with MC slopes and HLL fluxes; `order=1` retains
the reference method. Species-density slopes use the product rule after projecting composition
slopes onto sum(dX)=0. No composition renormalization or thermal-energy repair is applied.

EOS-inadmissible predictors have their cell slope zeroed, reported by `slope_reduced`.
If the full update remains inadmissible, a conservative flux limiter reduces a single
shared blend fraction between the high-order flux and the cell-centered HLL flux. This
preserves finite-volume budgets and species/mass consistency. `flux_fraction=1` means no
reduction; zero selects the low-order flux. The fraction is halved down to 2^-20, then
zero is tried. This deliberately simple global limiter can be diffusive. It checks the
actual EOS domain rather than assuming the ideal-gas pressure admissibility proof applies.
An inadmissible low-order update still refuses. The selected fraction is held fixed in
its piecewise derivative; a limiter regression checks finite reverse derivatives.

Periodic, outflow (zero-gradient), and left-reflecting/right-outflow planar boundaries are
available. Spherical geometry reflects at the origin and uses an outflow exterior. Shell
areas and volumes enter the conservative divergence, and the geometric pressure source
also enters the half-time predictor. Radial origin mass/energy fluxes vanish to roundoff.
The timestep bounds are dt*speed/dx <= 0.5 planar and <= 1/6 spherical. Invalid inputs,
EOS states or updates return `valid=False` and NaN states. A caller must inspect the mask.

`nuclear.coupling.make_step` uses half-burn / hydro / half-burn. Sources change species
and thermal energy at fixed momentum. Each half-burn refuses maximum absolute abundance
changes or fractional temperature changes above `burn_limit` (default 0.1).

Shock burning is suppressed by default. The 1D detector requires compression and
`abs((P[i+1]-P[i-1])/2 - rho*g*dx) / P > 2/3`. The second half-burn uses the union of the
first and newly detected shock masks. `burn_suppressed` records both masks; suppressed
cells take zero burn duration. This is a 1D adaptation of the pressure-jump/compression
criterion described in [Castro's reactions documentation](https://amrex-astro.github.io/Castro/docs/reactions.html),
not a claim of identical multidimensional detection. Smooth-flow refinement asserts
that neither shock suppression nor flux limiting intervenes.
Config fields `suppress_shock_burning` (default true) and `shock_threshold` (default 2/3)
set this detector, so suppression sensitivity runs need no code change.

`burn_min_temperature` and `burn_min_density` (default 0) skip the network in cells at or
below either value; `burn_skipped` records both half-burn masks. Arepo detonations use the
same kind of cut: T > 1e6 K in [Pakmor et al. 2022](https://arxiv.org/abs/2203.14990),
T > 2e7 K and rho > 1e4 g/cm³ in [Holas et al. 2025](https://arxiv.org/abs/2505.09445).
The aprox13 rates refuse T < 1e6 K (`src/luna/nuclear/light_rates.py:18`), so a cold
surface cell without a cut refuses every step. Low-resolution fronts are sensitive to
`shock_threshold` ([Brady & Zingale 2025](https://arxiv.org/abs/2505.07918)).

`gravity.spherical_1d` computes volume-averaged acceleration and potential, exact binding
energy for piecewise-constant shells, and potentials with zero at infinity. Symmetric
momentum kicks accompany a mass-flux-based gas-energy update. The gas-plus-gravity budget
includes the potential-energy flux of escaping/incoming mass. `boundary_flux` includes
this contribution when gravity is enabled. Spherical boundary budgets are per steradian;
multiply by 4*pi to compare with physical shell integrals.

## Running and restart

```bash
PYTHONPATH=src python main.py stellar /path/to/helm_table.dat \
  --config scripts/examples/stellar/he.json --stop-time 1e-5 \
  --dt-max 2e-6 --output out/stellar-he
# Other examples: co.json and white_dwarf.json in the same directory.
```

The hotspot examples initialize hot fuel against cold fuel with a reflecting left wall.
They are ignition/hydro smoke cases, not calibrated self-sustaining detonations.
The white-dwarf example starts an isothermal, self-gravitating C/O model. Its builder
integrates hydrostatic equilibrium and volume-averages shell quadrature. The ambient
cutoff is explicit (`ambient_density`, default rho=1e-4 g/cm³); no ongoing density or
temperature floor is applied. The raw hydro/gravity operator drifts from this initialization.
The example enables `well_balanced`: an explicit discrete residual-balancing method. Each
step subtracts the reference model's transport and momentum increment from the evolved
model's increment, and recomputes conservative gravitational work using the resulting
mass flux. This holds the supplied reference state exactly, while allowing perturbations
to evolve. It requires a physically checked reference: it can also hold an incorrect model.
The constant ambient is held by this residual correction too, so its motion is not a
validated freely falling atmosphere. Perturbation convergence and ambient sensitivity
remain necessary beyond a reference-hold test.

`white_dwarf_ignition.json` adds an artificial central thermal trigger to that C/O model.
For `r < ignition_radius`, the cell-center temperature is raised by
`(hot_temperature - T_initial) * (1 - (r / ignition_radius)^2)^2`; outside it is unchanged.
Density, species and momentum remain unchanged. The radius must span at least two cells
and lie inside the stellar surface; this minimum is an input check, not a resolution claim.
`hot_temperature` is the profile's extrapolated central peak, not the first cell's value.
The EOS supplies the new thermal energy. Diagnostics record the volume-integrated
`ignition_energy` in ergs as an external initial energy input, separate from nuclear release.
With balancing enabled, the reference remains the **unheated** model. Restarts preserve
that reference and the recorded input energy, and do not apply ignition again.
This example defines a controlled initial condition, not a validated explosion or ignition
mechanism. Ignition-size/temperature sensitivity and propagation checks remain required.
On the Intel Mac CPU with JAX 0.4.38/FP64, this 128-cell initialization passes EOS recovery,
keeps mass/composition/momentum and all cells outside the kernel unchanged, and adds
2.8196057e47 erg across six heated cells. Its recorded energy agrees with an independent
shell-volume sum to 1.1e-16 relative error.
[Initial-condition report](validation/white_dwarf_ignition_cpu.json).
Use `--config scripts/examples/stellar/white_dwarf_ignition.json --stop-time 0` with the
stellar command to construct and checkpoint the model without evolving it.

The coupled ignition smoke check uses a separate coarse 32-cell model with a 1.2e8 cm
kernel and advances only 4e-8 s, capped at 1e-8 s per timestep. It checks carbon consumption,
motion, shell-integrated mass and thermal+nuclear+gravitational energy budgets, preservation
of the cold balancing reference, and bitwise agreement with a halfway restart. Its larger
kernel permits three heated cells on this coarse grid; it is not the 128-cell example's
resolution study.

```bash
PYTHONPATH=src python tools/check_stellar_driver.py /path/to/helm_table.dat \
  --case white-dwarf-ignition --report stellar-ignition-smoke.json
```

This check passes on the Intel Mac CPU with JAX 0.4.38/FP64: four accepted steps without
retries, positive carbon consumption, mass-budget error 7.4e-18 and combined energy-budget
error 8.6e-17 relative to initial thermal plus absolute gravitational energy. The halfway
restart is bitwise identical and preserves the unheated reference and ignition energy.
The largest speed occurs in a dilute surface cell (rho about 8e-4 g/cm³); the innermost
cell moves at about 90 cm/s. This coarse surface response is not validated ejecta dynamics.
[Coupled ignition smoke report](validation/stellar_ignition_smoke_cpu.json).

The runner retries an entirely refused step from its unchanged input with half the timestep;
21 failed attempts abort. Accepted-step diagnostics count retries, slope reductions,
suppressed burns, minimum flux fraction and accumulated boundary budgets.
`--checkpoint-every N` adds intermediate checkpoints; initial and final states are saved.
`--max-steps` bounds the total accepted step count, including a restart's existing count.

```bash
PYTHONPATH=src python main.py stellar /path/to/helm_table.dat \
  --config scripts/examples/stellar/he.json --stop-time 2e-5 \
  --dt-max 2e-6 --output out/stellar-resumed \
  --restart out/stellar-he/state_00000006.h5
```

Use the actual checkpoint name printed by the first run. HDF5 stores conservative state,
temperature guesses, time, step count, diagnostics, configuration, the adaptive timestep
controller, schema version, and EOS table SHA256. Balanced checkpoints also store the
complete reference state. Restart refuses a different configuration/table or invalid state. Atomic
publication refuses existing checkpoint files. The stored state is lossless; identical
future timesteps are required for bitwise equality of continued trajectories.

## CPU verification

These checks use Intel Mac CPU, FP64 and JAX 0.4.38. They do not establish production
JAX 0.10 compatibility, GPU performance, or sharded execution.

- The nonreactive C/O acoustic test at 32/64/128 cells gives normalized density L1 errors
  0.001519 / 0.0002604 / 0.00005152 and velocity errors
  0.0009939 / 0.00009955 / 0.00001575. Both decrease by more than a factor of four over
  these grids; scaled conservation errors are below 1.6e-16.
  [Acoustic report](validation/stellar_acoustic_muscl_cpu.json).
- Joint reactive refinement uses a smooth periodic He/C mixture, rho~1e6, T~1e9,
  duration 0.001 s, and 16/32/64 cells at CFL 0.1. Observed orders are 1.97–2.33 for
  density, momentum, He, C and energy. The energy-minus-nuclear-binding invariant and
  species bounds pass. [Reactive joint report](validation/reactive_joint_cpu.json).
- The retained four-cell reactive timestep-only test converges at order 1.03. Hancock's
  cell-local predictor is not a second-order method-of-lines integrator at fixed coarse
  spatial resolution. It must not be used as the joint order test.
  [Fixed-grid report](validation/reactive_nonuniform_muscl_cpu.json).
- A stationary Mach-3 C/O shock solved from Rankine–Hugoniot/EOS conditions converges at
  32/64/128 cells with scaled L1 errors 0.00970 / 0.00512 / 0.00282 and open-boundary
  conservation errors below 1.2e-16. Discontinuities do not have a second-order L1 gate.
  [Shock report](validation/stellar_shock_muscl_cpu.json).
- The white-dwarf builder at rho_c=1e7, T=1e7 gives M~0.78907 solar masses and
  R~7.31666e8 cm. Relative virial residuals at 64/128/256 cells are
  7.32e-4 / 1.82e-4 / 4.53e-5, passing second-order refinement.
  [Initial-model report](validation/white_dwarf_cpu.json).
- The explicitly balanced 128-cell white dwarf preserves its supplied static reference
  bitwise in the hold check. A radial velocity perturbation evolved for two dynamical
  times (3.86794 s, 2310 steps): mass-weighted density change 2.36e-5, maximum core Mach
  number 0.00134, and gas-plus-gravity energy-budget error 1.74e-16. This is a single-grid
  stability test, not a converged oscillation-frequency measurement.
  [Perturbed-star report](validation/stellar_equilibrium_cpu.json).
- The He and C/O hotspot driver checks pass, including retrying refused steps. The He
  fixed-timestep split/restart trajectory is bitwise equal to the uninterrupted run. The
  retained raw (unbalanced) WD run conserves energy but changes some surface-cell densities
  by about 1% in 0.001 s; this motivated explicit reference balancing.
  [Driver report](validation/stellar_driver_cpu.json). A separate real-WD round trip checks
  lossless storage of its reference state as well as its evolved state and temperature.
  [Balanced restart report](validation/stellar_balanced_restart_cpu.json).
- The full EOS/nuclear suite passed 97 tests before the final runner/reference additions;
  two additional tests cover retry rollback/controller restart and reference hold with a
  non-frozen perturbation. Component checks cover gamma-law HLL limits, EOS energy fluxes, species closure with
  varying composition, flux-limit conservation and reverse derivatives, shock detection,
  a static uniform radial fluid, analytic uniform-sphere fields/self-energy, gravitational
  energy derivatives and exchange, radial gas-plus-gravity budgets, and HDF5 round trips
  with mismatch/overwrite refusal.

The earlier first-order reports remain as reference data:
[acoustics](validation/stellar_acoustic_cpu.json), [shock](validation/stellar_shock_cpu.json),
[uniform burning](validation/reactive_hydro_cpu.json), and
[nonuniform burning](validation/reactive_nonuniform_cpu.json).

```bash
PYTHONPATH=src python tools/check_reactive_refinement.py /path/to/helm_table.dat --report reactive-joint.json
PYTHONPATH=src python tools/check_white_dwarf.py /path/to/helm_table.dat --report white-dwarf.json
PYTHONPATH=src python tools/check_stellar_driver.py /path/to/helm_table.dat --report stellar-driver.json
PYTHONPATH=src python tools/check_stellar_equilibrium.py /path/to/helm_table.dat --report stellar-equilibrium.json
```

The next verification layer is described in [DETONATION.md](DETONATION.md): independently
compiled prescribed-ash CJ conditions and finite-rate steady profiles are available;
reactive-hydro propagation and self-sustaining eigenvalue gates are separate.

## Remaining science gates

The longer-run configuration, diagnostics and restartable Raven submission are described
in [STELLAR_EXPLOSION.md](STELLAR_EXPLOSION.md). No longer stellar explosion has been
demonstrated by the local smoke tests.

- Self-sustaining He/C/O detonation propagation, resolved reaction zones and independent
  CJ/ZND speed/profile comparisons; a burning hotspot is not sufficient.
- Grid/time convergence of the perturbed white-dwarf response and reference balancing,
  surface/ambient sensitivity, dual energy, and gravity/reactive joint convergence.
- Broader moving Riemann problems, rarefactions, nonlinear composition contacts, and
  validation of limiter/suppression effects on resolved burning fronts.
- Integration into the multidimensional production assembly, GPU/Pallas and named sharding.
