# Type Ia capability (plan)

Scope: white-dwarf explosions first (detonations, double detonations, mapped merger states); WD
mergers end to end are harder on a grid and not a near-term target. Open sources only: no FLASH
code, no laser / ICF / terrestrial-HEDP features. Evidence and file:line references:
`research_notes/Type Ia capability plan/` (helmholtz_eos, networks_nse_coupling,
double_detonation_setups, hydro_coupling; local).

## Stages, each gated before the next

1. Helmholtz EOS in JAX (standalone).
   - Forward evaluator implemented: `luna/eos/helmholtz.py`, tests and independent C++ references;
     see `HELMHOLTZ.md`. Safeguarded temperature inversion and round-trip/failure tests are implemented;
     the remaining physical sweeps, branch-join checks and GPU validation are still open.
   - Table: Timmes 541 x 201 (rhoYe 1e-12..1e15, T 1e3..1e13), biquintic Hermite free energy,
     ~18 MB fp64 on device, direct-index gather. dP/drho (Gamma1, c_s) from the separate bicubic
     table: a `custom_jvp`, not plain autodiff. Coulomb term: two branches, guard the kinks.
   - Off-table cells refuse (Timmes' `eosfail`), never clamp (AMReX clamps silently).
   - T(e): safeguarded Newton in ln T on the `chemistry/thermo.py` pattern; refuse on
     non-convergence.
   - Gates: Maxwell residuals at round-off and ~1e-5 interpolation error (Timmes & Swesty 2000);
     AMReX `eos_cell` (He, rho 1e6, T 1e9: p 7.33195768e22, Gamma1 1.511401882, c_s 3.328894e8);
     Castro exact-Riemann test 1 states; round trip T -> e -> T.
2. Networks (standalone, on the existing Rodas3 integrator).
   - All 31 distinct unscreened rate families and effective channel assembly implemented; a short
     self-heating Rodas3 baseline passes a Radau comparison (`NUCLEAR.md`). Screen5 and default
     sneut5 have compiled component references. Longer He/C/O trajectories and a tolerance ladder
     pass Radau comparisons; the He endpoint matches the upstream burn-cell CI benchmark.
     The complete RHS matches compiled upstream at 166 states. The Dursi/Timmes printed C/O
     ignition time is reproduced with EOS Coulomb disabled; the default enabled case is 9.7% later.
     The legacy EOS runtime setting remains unconfirmed. GPU validation remains open.
   - aprox13 from the AMReX Microphysics port (BSD-3, redistributed with Timmes' permission):
     13 species + energy, 34 rates, (a,p)(p,g) folded into (a,g) (guard `denom > 1e-30`).
     Screening screen5, neutrinos sneut5 (Microphysics defaults). Later aprox19 (Ye via
     p -> n and Ni-56 capture) and pynucastro-generated he-burn networks for thin He shells.
   - pynucastro (BSD-3) as the generator: its numba Python output is the template for a JAX
     backend (none exists).
   - Gates: Dursi & Timmes 2006 one-zone ignition (5.96e-2 s at rho 1e9, T 1e9, C/O 50/50);
     Microphysics `burn_cell` aprox13 (He, rho 1e6, 0.01 s: T0 1e9 -> added e 8.559157083e16 erg/g,
     T 1.516461213e9 K); ZND speeds (Khokhlov 1989: at rho=3e8, T=2e8, C/O 50/50,
     D_CJ=1.210e9 but the pathological self-sustaining D*=1.218e9 cm/s; match network/EOS).
3. Hydro coupling (touches the Riemann core: operator-level diff, full regression, a review
   before merge).
   - Standalone MUSCL–Hancock 1D Helmholtz HLL hydro, conservative species fluxes and Strang
     burning coupling implemented; CPU acoustic and stationary Mach-3 shock convergence,
     uniform one-zone agreement, and nonuniform reactive timestep checks in `STELLAR_1D.md`.
     The standalone runner also has conservative flux limiting, shock-burning suppression,
     spherical geometry/gravity, isothermal WD construction and HDF5 restart. Joint reactive
     and initial-WD virial refinement pass near second order. Explicit reference balancing
     holds the static WD and a perturbed model remains stable for two dynamical times.
     Detonation-speed gates, converged WD perturbations, broader Riemann/contact tests and
     multidimensional production integration remain.
   - HLLC/HLL with per-face Gamma1 and reconstructed rho*e (Castro HLL_solvers.H; Athena++
     hllc.cpp with PVRS speeds); today `fluxes.py:55-56, 73-74, 130-132` hard-code gamma.
   - The standalone predictor uses conservative EOS flux differences; production integration
     and dual energy (Castro eta1 = 1, eta2 = 1e-4).
   - Composition as passive scalars on all paths (today 3D Pallas only), consistent multi-fluid
     advection (Plewa & Mueller 1999; Castro normalize_species_fluxes) and sum X = 1.
   - Floors at the stellar surface (Katz et al. 2016: 1e-5 g/cm^3, 1e5 K), logged; positivity
     flux limiter; momentum sponge near the boundary.
   - Hydrostatic equilibrium: pressure-slope limiting about the HSE profile (Castro use_pslope)
     or a well-balanced scheme; also fixes the open polytrope drift.
   - Conservative gravity energy (Castro grav_source_type 4; Katz 2016: one-orbit |dE/E|
     4.8e-2 -> 4.4e-4).
   - Burning coupling: Strang with a burning dt limiter first, simplified SDC next (Zingale et
     al. 2022: Strang Ni-56 varied with tolerance, SDC did not); no burning in shocks (Castro
     detector |grad P . u|dx/P > 2/3 and div u < 0) and the Kushnir et al. 2013 limiter (f 0.1).
   - Tracers for post-processing (Seitenzahl et al. 2010: 384 nuclides; ~32 per axis).
   - Gates: Zingale & Katz 2015 stellar shock tubes (arXiv 1501.01923; exact solver Castro
     Util/exact_riemann); gamma-law results unchanged; a Helmholtz WD held in equilibrium; species
     conservation; energy to round-off with conservative gravity.
4. Detonation tests: prescribed-ash CJ conditions match compiled upstream routines within
   7.6e-9; independent Radau steady He/C/O profiles conserve jump conditions below 2.5e-14.
   These are overdriven reference profiles, not self-sustaining eigenvalues; see `DETONATION.md`.
   C/O reactive hydro passes the 128/256/512-cell translated-profile ladder; the finest
   front-speed error is 0.359%. The He ladder completed but fails its coarse energy-error reduction gate; further
   refinement is needed. Remaining: planar He detonation (Castro Detonation inputs at 1.1e6 g/cm^3);
   1D/planar C/O detonations against the ZND speeds.
5. Science ladder (3D, uniform 1024^3 ~ 24 km is adequate for 1-3):
   1. bare C/O WD central detonation (Shen et al. 2018);
   2. off-centre hot spot (Holas et al. 2025: Ni-56 0.178 / 0.484 / 0.719 Msun at 0.9 / 1.0 / 1.1);
   3. Castro `subchandra` 1.1 + 0.05 Msun double detonation (Zingale et al. 2024: Ni-56 0.652 Msun,
      converged to < 1 % from 40 to 5 km); its initial-model generator is open;
   4. thin He shells (<= 0.02 Msun) need ~4 km (AMR) and a he-burn network: refuse them with
      aprox13, which underestimates He burning (Shen & Moore 2014; Zingale et al. 2024);
   5. mapped merger states (Mehta et al. 2026); compare nuclear energy release (1.3e51 erg across
      codes) rather than Ni-56, whose cross-code spread is large (1.0 Msun: 0.38-0.58).

## Open decisions

- Licence of `helm_table.dat` (none stated on cococubed; only LANL's copy says CC-BY) before it,
  or a derived `.npz`, enters the repo; REACLIB terms (site behind a bot wall).
- Carry rho*e as an extra variable (Castro) or reconstruct gamma_e (FLASH-style).
- Go-ahead for the Riemann-core change (stage 3).
- Integrator: benchmark diffrax (Kvaerno3/5, KenCarp4, vmapped) against LUNA's batched Rodas3 at
  matched accuracy (DVODE references for networks 5 and 17, then aprox13) before stage 2; the
  Rodas3 choice rests on one MI300A measurement against a hand-written ESDIRK (docs/CHEMISTRY.md).

## Still to obtain (browser needed)

Timmes 1999 PDF (IOP bot check), REACLIB terms, Seitenzahl et al. 2009 (NSE), Colella & Glaz 1985,
Zhang & Shu 2010, Kaeppeli & Mishra 2014/2016.
