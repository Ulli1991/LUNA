# Standalone Helmholtz EOS

`luna.eos.helmholtz` evaluates prescribed density, temperature and composition in FP64 CGS.
Temperature inversion is available in `luna.eos.inversion`; the standalone hydro, burning and
spherical runner are described in [STELLAR_1D.md](STELLAR_1D.md).

## Use

```python
import jax
jax.config.update("jax_enable_x64", True)
from luna.eos.helmholtz import load_table, evaluate_checked

table = load_table("/path/to/helm_table.dat")
state = evaluate_checked(table, rho=1e6, temperature=1e9, mean_mass=4., mean_charge=2.)
```

`mean_mass = 1 / sum(X_i/A_i)`, `mean_charge = mean_mass * sum(X_i Z_i/A_i)`,
and `Ye = mean_charge/mean_mass`.
Inputs broadcast. Pressure is erg/cm³; energy is specific internal energy in erg/g;
entropy and cv are erg/g/K; sound speed is cm/s. Derivatives hold composition fixed.
The sound speed is nonrelativistic, as in Microphysics; this module does not enforce a relativistic limit.

`evaluate_checked` raises at the host boundary on invalid inputs or unstable/nonfinite results.
Use `evaluate` under `jax.jit` or `vmap`: it returns a per-cell `valid` mask and NaN thermodynamics
for invalid cells. Callers must inspect the mask. Bounds are inclusive in rho*Ye and T.
Clipping protects the index gather only; out-of-table states are never accepted as boundary values.
Load once outside jit and pass the table or close over it. No automatic downloads or global caches.

## Physics and derivatives

The electron/positron contribution uses biquintic Hermite interpolation of free energy and its
nine stored derivatives, on the standard 541 x 201 grid. Density compressibility comes from the
separate bicubic table, avoiding cancellation of free-energy derivatives in pair-dominated matter.
A pressure custom JVP uses that compressibility and analytic dP/dT, including the chain through Ye.
The table is treated as fixed data by this JVP; differentiating table entries is unsupported.
Higher derivatives and full parameter sensitivities have not been validated.

Ideal ions and LTE blackbody radiation are analytic. Coulomb corrections follow the two
Yakovlev-Shalybkov branches in Microphysics, enabled by default. As there, a correction that
would make pressure or energy nonpositive is disabled; `coulomb_disabled` reports this per cell.
These switches and the join near coupling parameter Gamma=1 are not smooth. There is no
radiation limiter, temperature floor, partial ionization, or ion-degeneracy model.

The implementation follows AMReX-Astro/Microphysics commit
`99b9b47cdb6d0d459deefeacdd4d78639bbd5f93`, `EOS/helmholtz/actual_eos.H` and
`actual_eos_data.H`, with its physical constants for reproducibility. The repository includes
the JAX implementation and numerical reference outputs; upstream source and table data remain external.

## Data and verification

The table is external pending the redistribution decision in `TYPEIA_PLAN.md`; neither it nor
a converted table is committed. The standard ASCII column order is
`F, F_d, F_T, F_dd, F_TT, F_dT, F_ddT, F_dTT, F_ddTT`, followed by three four-column tables.
The loader retains free energy and compressibility and validates all blocks for size/finiteness.

`tests/test_helmholtz.py` needs no external data: it checks polynomial reproduction, mixed
partials, Maxwell identities, Coulomb derivatives, the pressure JVP, batching/JIT, FP64 enforcement,
and invalid-state/boundary handling. A synthetic analytic free energy supplies the test table.

The independent physical reference uses the actual Microphysics C++ routines, compiled with
small AMReX type/macro stubs. The generator does not translate their physics into Python:

```bash
python tools/helmholtz_reference.py /path/to/Microphysics luna-reference-solutions/microphysics/helmholtz_eos/helmholtz_reference.json
PYTHONPATH=src python tools/check_helmholtz.py /path/to/helm_table.dat
```

The reference JSON records source/table SHA256 and 50 states: pure He, pure C and equal-mass C/O,
Coulomb on/off, sampled from rho=1e-4..1e10 and T=1e5..1e10 (plus the published eos_cell He case).
The C++ harness exercises forward evaluation only, without the AMReX driver or temperature inversion.
On the Intel Mac with JAX 0.4.38, batched JIT agrees within 1.1e-13 in pressure, 6.1e-14 in energy,
and 4.6e-13 in Gamma1. Worst cv difference is 1.5e-7 in a cold degenerate state at rho=9.78e8,
T=1.96e5: temperature derivatives suffer cancellation of much larger interpolated free-energy terms.
The checker uses separate, explicit bounds per quantity; this comparison is implementation agreement,
not a test of accuracy against direct Fermi integrals. GPU/JAX 0.10.x checks remain to do.

Still needed for stage 1: a broader physical stability/consistency sweep and validation at the
Coulomb branch join. No stellar or detonation calculation is justified by the standalone EOS alone.

## Temperature inversion

```python
from luna.eos.inversion import temperature_from_energy_checked

result = temperature_from_energy_checked(table, rho=1e6, energy=state.energy,
                                         mean_mass=4., mean_charge=2.)
print(result.temperature)
```

The device function `temperature_from_energy` broadcasts inputs, uses Newton steps in log T,
and bisects when a step would leave the bracket or exceed half its width. Default bounds are
the table endpoints; `temperature_bounds=(lower, upper)` can restrict a known physical branch.
A positive finite `temperature_guess` is an initial hint and may lie outside those bounds.
The solver never clamps the requested energy to make an answer. Invalid endpoints, an unbracketed
energy, an invalid trial state or exhaustion of `max_iterations` produce `converged=False` and NaN
temperature. The checked wrapper raises if any cell fails. Inspect convergence when calling under jit.

Convergence requires both `|delta e| / (T cv) <= temperature_rtol` (default 1e-8) and a relative
energy residual <= `energy_rtol` (default 1e-12). Iteration counts and the final residual/correction
are returned per cell. Completed cells stay frozen while other cells continue.

Cold degenerate matter can lose temperature information in FP64 energy. A zero energy residual
therefore does not guarantee an equally accurate temperature. `relative_energy_resolution` reports
`eps |e| / (T cv)` as a conditioning diagnostic, not an error bound for the interpolant.
The 50 reference-state round trips recover T within 8.2e-9 relative. A separate 9 x 9 pure-carbon
sweep over rho=1e4..1e10 and T=1e5..1e10 converged in all 162 Coulomb-on/off cases; worst error was
4.2e-8 at rho=1e10, T=1e5 with Coulomb off (zero residual), and 4.3e-9 with Coulomb on.
These are CPU JAX 0.4.38 measurements, not GPU validation.

The Coulomb fit and its disable switch can make global inversion nonunique or discontinuous.
Choose a continuous, stable energy-increasing branch when restricting bounds. Failed convergence
near these changes is reported rather than hidden with a floor. The solver does not search for
all roots. Forward-mode differentiation is tested at a smooth interior state; reverse-mode through
the adaptive while loop is unsupported. An implicit derivative and broader sensitivity tests remain open.

## Broader validation and Raven

`tools/check_helmholtz.py TABLE --sweep --report report.json` checks a 9 x 9 grid in density
and temperature for He, C and equal-mass C/O, with Coulomb corrections on/off (486 states).
It requires positive cv/compressibility, finite valid results, normalized Maxwell residuals <= 1e-12,
and converged temperature recovery. Round-trip errors above 2e-8 are recorded as precision limited;
the acceptance allowance is 64 times the energy-resolution diagnostic in addition to 2e-8.
This is a numerical tolerance for this sweep, not a rigorous accuracy bound.

The branch tests retain the discontinuity at Gamma=1: approximately 7.0e-6 in the Coulomb correction's
energy and 5.36% in its entropy. These are relative to the correction, not to the total EOS quantities.
Maxwell relations are tested separately on either side; a derivative at the join is not asserted.

From the repository root on Raven, after pushing these changes and providing the external table:

```bash
sbatch submit/examples/helmholtz_raven.sbatch /path/to/helm_table.dat
```

The job runs the reference comparisons, sweep and standalone unit tests. `--require-gpu` refuses CPU
fallback; the JSON records JAX version and device identity. This job has been prepared, not executed
from the local Mac. A successful local sweep does not certify the entire Helmholtz table or close the
stellar shock-tube and GPU gates.
