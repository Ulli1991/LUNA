# Detonation verification

The verification path separates prescribed-ash CJ conditions, finite-rate steady profiles,
and time-dependent reactive hydro. A burning hotspot alone does not pass these gates.
All current measurements use Intel Mac CPU, JAX 0.4.38 and FP64.

## Prescribed-ash CJ

`nuclear.detonation.prescribed_ash_cj` solves the reactive Hugoniot and frozen-sonic condition
with an explicitly supplied ash composition. It refuses invalid/nonconvergent or subsonic
branches. It is not a chemical-equilibrium or pathological-detonation eigenvalue solver.

`tools/detonation_reference.py` compiles the external pinned Microphysics Helmholtz EOS
and its `util/cj_detonation/cj_det.H` routines. No upstream source is copied into LUNA.
The numerical fixture covers He and C/O fuel at several densities with Coulomb corrections
both disabled and enabled, with prescribed pure Ni56 ash. The independently solved JAX/SciPy
results agree within 7.6e-9 relative error across ten states. A separate synthetic-EOS test
reproduces the analytic ideal-gas CJ speed.

[Fixture](../luna-reference-solutions/microphysics/detonation_cj/detonation_cj_reference.json),
[CPU comparison](validation/detonation_cj_cpu.json).

The plan's [Khokhlov 1989](https://articles.adsabs.harvard.edu/pdf/1989MNRAS.239..785K) value 1.210e9 cm/s is not a general self-sustaining C/O target.
For its rho=3e8, T=2e8, C/O 50/50 example, the pathological speed is 1.218e9 cm/s.
Its reaction network, equilibrium ash and EOS settings must be matched before making a
quantitative comparison. In particular, replacing its ash by fixed pure Ni56 changes the
problem substantially: the fixed-ash calculation here gives about 1.430e9 cm/s.

## Finite-rate steady flow

`nuclear.detonation.make_znd` starts from a frozen-composition shock at a prescribed speed.
For each composition it solves steady mass, momentum and total-enthalpy constraints,
including the composition-dependent binding energy. An implicit first derivative through
that solve provides the Jacobian for the independent SciPy Radau integration. The sound
speed remains the EOS frozen-composition acoustic speed. Choked, nonconverged or off-table
states refuse; this implementation is for overdriven profiles, not eigenvalue selection.

Internal Radau/Newton trial abundances admit a signed algebraic extension up to 1e-6;
accepted reference profiles are checked separately for negative traces below -1e-12,
species sum, EOS validity and jump residuals. No trial-state extension is applied to the
hydro solver's accepted states.

The initial benchmark speeds are 1.2 times the prescribed-Ni56 CJ speeds, with screen5 and
no neutrino cooling. This factor labels a reproducible overdriven case; it does not establish
a factor relative to the network's actual self-sustaining speed.

| Fuel | Upstream density | Speed (cm/s) | Primary-fuel half-consumption length (cm) |
| --- | ---: | ---: | ---: |
| Pure He | 1e6 | 1.847916616e9 | 1.5920544e6 |
| C/O 50/50 | 1e7 | 1.478911687e9 | 0.01184816 |

Both use upstream T=1e8 K. Maximum normalized jump residuals along the computed tails are
below 2.5e-14. The C length is not the full oxygen/silicon reaction-zone length.
[He profile](validation/znd_he_tail_cpu.json), [C/O profile](validation/znd_co_tail_cpu.json).

## Reactive-hydro comparison

`tools/check_detonation_hydro.py` initializes cell averages by three-point quadrature of a
steady profile, splitting the integration interval exactly at the moving shock, and follows
its translation through a planar domain. The discontinuity must not be treated as a smooth
quadrature interval. Final conservative states are saved alongside each report so the
comparison can be checked without repeating the evolution. Trace projection at
the IC boundary is explicitly reported and refuses corrections above 1e-12. The run keeps
shock-burning suppression and flux limiting enabled, counts retries, and compares the
measured front speed and near-front profile with the translated steady solution. The
fine-grid speed gate is 3%; density, primary-fuel and energy L1 errors must decrease
with refinement. Global mass and energy-minus-binding budgets are also checked.

The domain spans 40 primary-fuel half-consumption lengths, with the shock initially near
one third of the domain. It advances four such lengths. These are short overdriven-profile
checks; they are not a demonstration of a self-sustaining wave independent of its rear state.
The profile must cover the whole reactive portion of the domain. Boundary contamination,
induction/reaction-zone resolution and shock-burning suppression require separate checks
before extrapolating to self-sustaining detonations.

```bash
PYTHONPATH=src python tools/check_detonation_reference.py /path/to/helm_table.dat --report cj.json
PYTHONPATH=src python tools/check_znd.py /path/to/helm_table.dat --case he --tail --report znd-he.json
PYTHONPATH=src python tools/check_detonation_hydro.py /path/to/helm_table.dat znd-he.json --report hydro-he.json
PYTHONPATH=src python tools/check_znd.py /path/to/helm_table.dat --case co --tail --report znd-co.json
PYTHONPATH=src python tools/check_detonation_hydro.py /path/to/helm_table.dat znd-co.json --report hydro-co.json
```

Reports named `*_legacy_cpu.json` retain earlier runs whose profile comparison used
unsplit quadrature across the shock. They are marked superseded and do not establish
profile convergence.

The corrected C/O ladder passes at 128, 256 and 512 cells (3.2, 6.4 and 12.8 cells
per carbon half-consumption length). On the finest grid the front-speed error is 0.359%;
normalized density, primary-fuel and energy L1 errors are 0.03581, 0.01812 and 0.01869.
All three decrease at each refinement. No timesteps were retried, and the maximum
normalized mass/energy-budget errors across the ladder are below 7e-16.
[C/O hydro report](validation/detonation_hydro_co_cpu.json).

The He 64/128/256-cell run is complete but **does not pass the declared ladder gate**.
All three profile errors decrease monotonically, but the energy L1 ratio from 64 to 128
cells is 0.8155, exceeding the required 0.8. From 128 to 256 cells, density, fuel and
energy error ratios are 0.4890, 0.5998 and 0.4865. The finest-grid speed error is 1.161%,
with one timestep retry and normalized conservation errors below 2.4e-15 across the ladder.
The very small 128-cell speed error was not maintained at the next refinement; it should
not be interpreted as an accuracy estimate. The Mac 512-cell attempt was stopped after the 11% progress checkpoint because of
CPU cost; it produced no completed-grid result. The completed 64/128/256-cell results
are preserved. The next ladder is 128/256/512 cells on Raven, with the same 0.8
profile-error ratio and 3% speed gates. New snapshots also store the front-position
time history. Snapshot-displacement estimates across several
gradient windows decrease in error with refinement, but differ from the late-time speed
fit; the near-perfect 128-cell fit is therefore not evidence of near-perfect propagation.
[Front diagnostic](validation/detonation_he_front_diagnostic_cpu.json).
[He hydro report](validation/detonation_hydro_he_cpu.json).

## Raven handoff

From the repository root in the prepared Raven checkout:

```bash
sbatch --partition=YOUR_GPU_PARTITION submit/examples/detonation_raven.sbatch /path/to/helm_table.dat
```

Choose a GPU partition allowing the requested two-hour wall time. The job uses the
same module and environment setup as the existing Raven examples (`~/jax-env`),
requires SciPy in that environment, and refuses a CPU backend. The external Helmholtz
table must match the saved reference hash. All three grids run on Raven to establish
convergence on the same software/device stack. The job writes uniquely named reports,
per-grid snapshots and front histories without replacing the Mac results. The saved
steady profile can be reused; its table hash is checked by the hydro tool. This is a
validation run, not a GPU performance benchmark. The job has not been submitted here.

## Remaining

- Resolving the failed He ladder gate with further refinement and front tracking checks.
- Establishing network-matched self-sustaining eigenvalues, including pathological C/O
  branches and equilibrium versus frozen sound speeds.
- Sustained propagation, boundary/domain-length studies, ignition-independence checks,
  and comparisons with matched published ZND profiles and speeds.
