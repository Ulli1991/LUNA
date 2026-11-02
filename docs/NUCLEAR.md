# Nuclear network development

The first target is a standalone aprox13 self-heating burn, then a fixed-grid 1D planar detonation,
then a spherical white-dwarf explosion. Hydro does not call these modules yet.

## Implemented

`luna.nuclear.aprox13` defines 13 alpha-chain species and 17 effective reaction channels.
`effective_rates` assembles all unscreened channels, including carbon/oxygen fusion branches and
eight equilibrium proton-mediated paths. The cutoff for an inactive proton intermediate includes
density, as in the reference implementation. `reaction_flows` applies density powers, molar
abundances and identical-particle factorials once. Coefficients have no density or abundance factors.

`derivatives_from_flows` returns mass-fraction derivatives and specific heating in erg/g/s.
Increasing positive binding energy increases thermal energy. For mass- and charge-conserving
reactions this avoids subtracting large nucleon rest masses. Abundance products avoid generic
power derivatives at zero, keeping the Jacobian finite for absent species.

`light_rates` and `alpha_rates` supply all 31 distinct unscreened raw coefficient families used by
these channels. Their supported temperature interval is 1e6..1e10 K; other inputs return NaNs.
The carbon alpha capture prescription is 1.7 times CF88, not the optional deBoer fit. The generic
heavy fits combine prefactors and barriers in log space to avoid intermediate exponential underflow.
Two-body forward coefficients have units cm³/mol/s, triple-alpha forward cm⁶/mol²/s, and
photodisintegration coefficients 1/s. Proton exchange reverse coefficients are also two-body rates.

`burn.make_network(table, screening="none", neutrinos="none")` connects the 13 mass fractions plus
specific thermal energy to Rodas3. Temperature comes from Helmholtz inversion, with an implicit
first derivative for the RHS Jacobian. Reuse the network object to reuse its compiled closure.
Screening and neutrino options are mandatory: `none`/`screen5` and `none`/`sneut5`, respectively.
Other requested models raise; there is no silent substitution. Cooling subtracts the specific
neutrino loss from thermal heating without changing the instantaneous species derivatives.

`screen5` evaluates weak, intermediate and strong screening, capped at exp(300). As in the pinned
aprox13 implementation, enhancement factors multiply both forward and reverse raw rates before
folding the proton-mediated channels; triple-alpha uses the product of two entrance factors.
The 172-state compiled reference covers the screening transitions.

`sneut5` includes pair, plasma, photo and bremsstrahlung losses, matching the reference default
with recombination disabled. Requesting optional recombination raises. It returns zero below
1e7 K and NaNs for invalid states. Individual signed bremsstrahlung fit values are retained;
the total must be finite and nonnegative. All four components agree with a compiled external
reference at 315 states (He, C and C/O; rho=10..5e9 g/cm³, T=1e6..1e10 K), including temperature
switches, within 1e-12 relative error. Smooth derivatives and the cooling contribution are tested.

The public composition functions reject negative or unnormalized abundances by default. Internal
Rodas3 stages allow signed trace values down to -1e-14 without clipping the RHS polynomial.
This avoids rejecting stages solely because an absent species has round-off-level negativity.
Accepted steps still pass the integrator's bounds check (which clips trace negatives within its
absolute tolerance); final mass and energy budgets must be checked. Larger negative stages refuse.
This extension is tested separately. Full burn sensitivities and reverse-mode gradients are not validated.

## References and verification

Constants and legacy fit data follow Microphysics commit `99b9b47cdb6d0d459deefeacdd4d78639bbd5f93`.
The full reference generator explicitly restores missing forward assignments in `rate_cr48ap` and
`rate_fe52ap`. Both assignments are present in the current upstream rate source, checked 2026-10-14:
https://github.com/AMReX-Astro/Microphysics/blob/main/rates/aprox_rates.H
The numerical JSON records these corrections; it does not claim to be an unmodified old checkout.
Upstream source remains external. Local JAX names describe their roles or physical quantities.

`tests/test_aprox13.py` checks conservation, forward/reverse symmetry factors, heating signs,
composition, zero-abundance Jacobians, branching, density cutoffs, batched JIT, the implicit thermal
derivative and stage behavior. All 31 raw families are compared against compiled C++ routines at
39 temperatures, including either side of temperature switches. Relative rate bounds are 3e-12;
the full comparison uses an absolute floor of 1e-270 for negligible tails where the reference's
intermediate exponentials lose subnormal precision. Smooth temperature derivatives are also checked.
A Rodas3 regression uses an analytic ions+radiation EOS; it is not a WD reference model.

```bash
python tools/nuclear_rate_reference.py /path/to/Microphysics luna-reference-solutions/microphysics/nuclear_rates/nuclear_all_rates.json --full
PYTHONPATH=src python -m pytest -q tests/test_aprox13.py
PYTHONPATH=src python tools/check_nuclear_burn.py /path/to/helm_table.dat --screening screen5 --neutrinos sneut5 --scipy
```

The external-table check burns initially pure He at rho=1e6 g/cm³, T=1e9 K for 1e-4 s. With
screening/neutrinos explicitly off, the Intel Mac CPU, JAX 0.4.38 result is T=1.004823787e9 K,
X(He)=0.9989306363, 34 accepted steps and 5 rejections. The relative thermal-versus-binding-energy
budget residual is 2.8e-12. It agrees with a separate SciPy Radau integration of the same RHS at
rtol=1e-9 (abundance comparison: rtol 5e-5, atol 1e-12). This tests the integration, not independent
nuclear physics; rate comparisons and reaction checks supply separate verification.

The screened short burn without neutrinos also passes Radau: T=1.005262503e9 K,
X(He)=0.9988331733, 36 accepted steps and 5 rejections; energy residual 3.9e-12.
The verification tool integrates escaped neutrino energy as an extra Radau diagnostic when
cooling is enabled, checking thermal plus escaped energy against binding-energy release.
This diagnostic is separate from the 14-variable production state.
With both screen5 and sneut5 enabled, the same Intel Mac/JAX 0.4.38 short test passes:
36 accepted steps, 5 rejections, T=1.005262503e9 K and X(He)=0.9988331733.
The Radau thermal-increment difference is 1.2e-12 relative; thermal plus escaped-neutrino
energy closes against binding-energy release to 1.3e-12 relative. Escaped energy is only
7077 erg/g here, so this is not a cooling-dominated benchmark. The local targeted suite
passes 29 tests (the slow Rodas3 regression excluded); GPU and production-JAX checks remain open.

## Longer He and C/O burns

`tools/check_burn_trajectories.py` checks constant-density self-heating burns with screen5 and
sneut5 enabled. It samples complete abundance/energy states at logarithmic checkpoints, compares
with Radau at rtol=1e-10, and runs Rodas3 at rtol=1e-5, 1e-6 and 1e-7. Checkpoints restart the
integration with the previous composition, energy and temperature guess. Radau starts with a
1e-12 s step to keep its initial error-estimation trial within the signed-trace allowance.
No negative reference abundances are clipped to make the comparison pass.

The Intel Mac CPU / JAX 0.4.38 FP64 results at Rodas3 rtol=1e-7 are:

| Initial fuel | Density (g/cm³) | Initial T (K) | Duration (s) | Final T (K) | Max relative heating difference from Radau |
| --- | ---: | ---: | ---: | ---: | ---: |
| He | 1e6 | 1e9 | 0.01 | 1.51646123e9 | 5.1e-10 |
| C/O 50/50 | 1e9 | 1e9 | 0.06 | 1.12971914e9 | 8.1e-8 |
| C/O 50/50, hot | 1e7 | 2e9 | 0.01 | 4.43890257e9 | 1.2e-7 |

All three improve as tolerance tightens. The hotter C/O case consumes almost all its initial
carbon and oxygen and becomes silicon/sulfur rich; it takes about 5900 accepted steps at the
tightest tolerance. These are integration/consistency checks, not an independent C/O ignition
benchmark or an NSE-equilibrium validation. Comparison gates require abundance L1, thermal-
increment and temperature errors below 100 times Rodas3 rtol, and the relative energy-budget
residual below 1e-7. Heating and budget denominators have a 1e8 erg/g floor for tiny increments.
The energy budget uses escaped-neutrino energy integrated by the independent Radau solve.

The apparently large dense C/O discrepancy at 0.06 s comes from stopping just before the
runaway. A dedicated event solve to 90% carbon consumption isolates the EOS Coulomb correction:

| Physics change from baseline | Ignition time (s) |
| --- | ---: |
| Screen5, sneut5, EOS Coulomb enabled | 0.0653897074 |
| Neutrinos disabled | 0.0653897071 |
| EOS Coulomb disabled | 0.0596257053 |
| Screening disabled | 2.59169411 |

[Dursi & Timmes, Table 1](https://arxiv.org/pdf/astro-ph/0510367) gives 0.0596 s for rho=1e9 g/cm³,
T=1e9 K, C/O 50/50, defining ignition by 90% carbon consumption. Disabling the EOS Coulomb
correction reproduces that time at its printed precision (absolute comparison bound 5e-5 s).
The paper's exact legacy EOS option has not been independently recovered; this is evidence for
the source of the timing shift, not proof of their runtime setting. The default with EOS Coulomb
enabled is 9.7% later; it is retained. Neutrino losses cannot explain that shift.

`burn.initial_state(..., coulomb=False)` and `burn.make_network(..., coulomb=False)` select the
comparison consistently in the initial EOS, temperature inversion and RHS EOS evaluation.
Both default to True. The diagnostic uses Radau rtol=1e-9 and includes escaped-neutrino energy;
all four event runs conserve thermal plus escaped energy against binding release within 1e-13
relative. Event temperatures are about 5.8e9 K. See [the ignition report](validation/carbon_ignition_cpu.json).
Tightening Radau rtol from 1e-9 to 1e-10 changes the Coulomb-disabled ignition time by
5.6e-14 s; the [tight run](validation/carbon_ignition_tight_cpu.json) includes the automated
comparison with the paper's printed precision. The targeted nuclear suite passes 31 tests
(the slow Rodas3 regression excluded), on Intel Mac CPU/JAX 0.4.38.

The helium endpoint also matches the pinned Microphysics Rosenbrock `burn_cell` CI benchmark:
T=1.516461213e9 K and added energy=8.559157083e16 erg/g. The fixture records the source hash,
input file, CI temperature override and comparison tolerances. Its upstream abundance atol is
1e-6; this comparison uses rtol=1e-5, atol=1e-8 and does not validate relative trace abundances.
The quoted upstream `Hnuc=8.559157083e18` is the average heating rate, not the integrated energy.

Full states, reference temperatures, tolerance errors, step counts, table hash and source hashes
are in [the CPU report](validation/typeia_burns_cpu.json). Timing fields include compilation and
host diagnostics; they are not GPU performance measurements.

```bash
PYTHONPATH=src python tools/check_burn_trajectories.py /path/to/helm_table.dat --report burn-trajectories.json
# Optional: --case helium, carbon_oxygen, or hot_carbon_oxygen
```

## Complete upstream RHS comparison

`tools/nuclear_rhs_reference.py` compiles the external network's reaction definitions, density
scaling, screening, branching, proton-path postprocessing, identical-particle factors, species RHS
and rest-mass heating calculation. The harness uses double precision with rate tabulation and
neutrino cooling disabled. It replaces compile-time screening-pair calculation with runtime
evaluation of the same formulas and applies the two documented missing-forward-assignment fixes.
No upstream source is stored in LUNA. Original header hashes and harness settings accompany the
numeric fixture.

All 166 states pass, including the He/C/O trajectory checkpoints, mixtures spanning all species,
rho=1e4..1e9 g/cm³ and T=1e8..8e9 K, with screening both on and off. The absolute comparison
bound is 5e-12 times gross forward-plus-reverse turnover, plus 1e-250 for negligible tails.
Using turnover avoids falsely rejecting nearly cancelled net derivatives near equilibrium;
the upstream rest-mass heating sum also loses digits relative to the binding-energy formulation.
The EOS Coulomb toggle is separately covered by the implicit-temperature derivative test.

```bash
python tools/nuclear_rhs_reference.py /path/to/Microphysics luna-reference-solutions/microphysics/aprox13_rhs/aprox13_rhs_reference.json
PYTHONPATH=src python tools/check_carbon_ignition.py /path/to/helm_table.dat --report carbon-ignition.json
# Optional: --case no_eos_coulomb --rtol 1e-10
```

## Extended CPU burn coverage

The extended trajectory suite adds dilute helium (rho=1e5), hot helium (rho=1e6), carbon-rich
70/30 C/O (rho=1e7), oxygen-rich 30/70 C/O (rho=1e8), and pure nickel relaxation (rho=1e7).
Initial temperatures span 2e9..5e9 K. Each passes Rodas3 rtol=1e-5, 1e-6, 1e-7 comparisons
against Radau, with abundance bounds, mass and thermal-plus-neutrino energy checks. A separate
10-second helium tail consumes 95.7% of the initial helium, reaching X(Ni)=0.9568 and
T=3.58112e9 K. It still has residual burning; this is not complete helium exhaustion.

| Case | Duration (s) | Final T (K) | Max heating difference at rtol=1e-7 |
| --- | ---: | ---: | ---: |
| Dilute helium | 0.1 | 2.00948e9 | 3.6e-9 |
| Hot helium | 0.1 | 3.41552e9 | 7.5e-9 |
| Carbon-rich C/O | 0.1 | 4.62052e9 | 1.1e-8 |
| Oxygen-rich C/O | 0.01 | 6.38995e9 | 2.2e-8 |
| Nickel relaxation | 0.01 | 4.92591e9 | 1.3e-8 |
| Helium late tail | 10 | 3.58112e9 | 2.9e-9 |

[Extended trajectories](validation/extended_burns_cpu.json) and the
[helium tail](validation/helium_exhaustion_cpu.json) retain complete reference and solver states.
Every tolerance ladder improves. For tiny early energy increments the budget gate includes
an explicit roundoff allowance: 64 epsilon times the magnitudes of the initial/final stored
energies and composition-weighted binding energies. It retains the 1e-8 reference and 1e-7
Rodas3 relative residual bounds; reports include both raw residuals and excess above roundoff.
This avoids treating subtraction of large stored energies as an accurately resolved tiny release.

The [reaction-balance diagnostic](validation/extended_burn_balance_cpu.json) distinguishes stiff
forward/reverse balance from simple fuel depletion. The oxygen-rich and nickel runs have
summed absolute net-to-gross reaction-flow ratios 2.3e-7 and 1.4e-8, respectively. Nickel
relaxation also checks endothermic photodisintegration. These exercise near-balanced kinetics;
they do not establish full physical NSE, weak reactions or evolving Ye. The helium tail remains
out of balance (ratio 0.024), despite substantial fuel consumption.

```bash
PYTHONPATH=src python tools/check_burn_trajectories.py /path/to/helm_table.dat --extended --report extended-burns.json
PYTHONPATH=src python tools/check_burn_balance.py /path/to/helm_table.dat extended-burns.json --report burn-balance.json
```

A standalone MUSCL–Hancock Helmholtz hydro and split-burning prototype is now available;
see [the 1D hydro status](STELLAR_1D.md). The dedicated `main.py stellar` runner adds
shock-burning suppression, conservative flux limiting, spherical self-gravity and HDF5
restart. Joint reactive refinement passes at about second order; multidimensional
production assembly and detonation-speed validation remain.

## Remaining

- Published one-zone trajectories and ignition diagnostics at matched network/physics settings;
  the legacy EOS Coulomb setting remains unconfirmed despite the matched printed C/O time.
  The unscreened baseline must not be compared to screened burn-cell goldens as if identical.
- Independent equilibrium/NSE references, broader composition/thermal derivative validation,
  and GPU/production-JAX checks.
- Strong-shock, composition-contact and nonuniform reactive validation of the standalone 1D
  prototype; higher-order reconstruction, production integration and a planar detonation.
- Spherical geometry and an equilibrium, self-gravitating WD model after the planar checks pass.

A running short one-zone baseline is not a validated ignition calculation or detonation model.
