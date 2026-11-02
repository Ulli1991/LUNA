# What LUNA claims, and the test behind each claim

Statuses:
- **validated**: a test asserts the claim against an independent reference (exact or
  analytic solution, published data, another code's output, round-off/bitwise identity),
  with a stated bound.
- **shown**: real runs, shown in figures or measured; not (yet) asserted against an
  independent reference. The note says what a test does check, if anything (for example
  "regression only": a test pins LUNA's own result, so a change is caught but correctness
  is not shown).
- **experimental**: under development, not finalised (radiation, chemistry, sinks and the
  Torch coupling, block AMR). These modules have tests; they do not yet meet the standard
  of the hydro core.

Statuses follow the October 2026 test audit, re-checked against `main`. Test locations are
`file:line` of the test function on `main` (2026-10-30); bound sources are in each test's
docstring. GPU-only tests and tests that need 2 or 8 devices run in the GPU and `--devices`
tiers, not in the CPU fast tier (docs/TESTING.md).

## Hydrodynamics

| claim | where stated | test | status | note |
|---|---|---|---|---|
| Euler equations in 1D, 2D, 3D, unsplit MUSCL-Hancock with PLM; second order on smooth flows | README | tests/test_hydro_accuracy_smooth.py:160, tests/test_hydro_accuracy_linear_waves.py:231, tests/test_regressions.py:105 | validated | |
| HLLC (default) and HLL fluxes match their textbook definitions and Toro's tests | README | tests/test_hydro_accuracy_riemann.py:67, :106, :128 | validated | |
| Shock tubes (Toro 1-5) converge to the exact Riemann solution | README | tests/test_hydro_accuracy_shocktube.py:157 | validated | |
| `riemann_solver="hll"` is the robust fallback: HLL keeps grid-aligned strong shocks planar where HLLC carbuncles | README, GALLERY | tests/test_hydro_hard_carbuncle.py:121 (hll cases pass, hllc cases strict xfail), :143 | validated | chosen by the user, never switched automatically (no automatic switch in src/) |
| Sedov-Taylor blast against the exact solution | README | tests/test_hydro_hard_sedov_offaxis.py:219, :243 (2D cylindrical, off-axis) | validated | |
| 3D Sedov shock radius and profile | README figure, GALLERY | tests/test_hydro_accuracy_sedov.py:69 | shown | regression only (scalar checks); the "<1 %" in GALLERY is read off a plot |
| Noh (planar, 2D, 3D) and Guderley implosion against exact solutions | README | tests/test_hydro_hard_noh_planar.py:141, tests/test_hydro_hard_noh_multid.py:286, tests/test_hydro_hard_guderley.py:345 | validated | |
| Linear growth of KH, Rayleigh-Taylor, Richtmyer-Meshkov against linear theory | README | tests/test_hydro_accuracy_kh.py:87, tests/test_hydro_hard_rayleigh_taylor.py:191, tests/test_hydro_hard_richtmyer_meshkov.py:218 | validated | |
| Nonlinear KH (McNally et al. 2012) against the Pencil Code reference | README | tests/test_hydro_hard_mcnally_kh.py:288, :457, :468 | validated | M(t) against the Pencil curve; rho and V_y at t = 1.5 against AthenaK PPMX 4096² (LUNA's own 4096² kept as regression only); LUNA van Leer is within 5 % of AthenaK PLM, 3-4 % above it in V_y (not "no less accurate") |
| KH at 8192² next to AthenaK | README figure, GALLERY | none | shown | accuracy comparison by eye, not an asserted gate; GALLERY's "converged to AthenaK" is stronger than the evidence |
| Shu-Osher, Woodward-Colella, Titarev-Toro | GALLERY | tests/test_hydro_hard_shu_osher.py:129, :137, tests/test_hydro_hard_woodward_colella.py:148, :156, tests/test_hydro_hard_titarev_toro.py:128, :136 | validated | gated against AthenaK PPMX + RK4 at 16384 / 16384 / 65536 cells (L1, L2 of rho at the tests' N) and LUNA van Leer against AthenaK PLM at the same N; LUNA's own fine-grid reference kept as regression only |
| Liska-Wendroff implosion (early time), Kurganov-Tadmor 2D Riemann configuration 3 | TEST_PLAN | tests/test_hydro_hard_implosion.py:400, :410; tests/test_hydro_hard_kt_riemann.py:291 (CPU), :298 (GPU), :446 | validated | against AthenaK PPMX 6400² (implosion t = 0.045) and 8192² (KT 3 outside the core); implosion late time, KT cores and KT configuration 6 regression only (no converged reference in either code); KT strips against the exact 1D solution |
| Cloud-crushing blob (Agertz et al.) | TEST_PLAN | tests/test_hydro_hard_agertz_blob.py:109 | shown | regression only |
| Polytrope in self-gravity held at rest | TEST_PLAN | tests/test_hydro_hard_well_balanced.py:110 | shown | regression only; the scheme is not well balanced (5-23 % central drift over 10 crossings) |
| Driven turbulence (OU forcing) | README, GALLERY | tests/test_turb_spectrum.py:43 (initial spectra, validated); tests/test_ou_forcing_split.py:71, :95 | shown | the forcing-split tests never run the forcing stage |

## Devices, kernels, runs

| claim | where stated | test | status | note |
|---|---|---|---|---|
| A run split over several devices gives the single-device result | README, CLAUDE.md | bitwise: tests/test_hydro_hard_sedov_offaxis.py:262 (2x4 mesh), tests/test_hydro_hard_mcnally_kh.py:307; to < 1e-12: tests/test_seam_consistency.py:116 | validated | |
| Pallas hydro kernel is the XLA kernel | README | tests/test_hydro_pallas_matrix.py:37 (bitwise on GPU, every dim/limiter/solver/precision), tests/test_hydro_pallas.py:403 | validated | the full production step on GPUs (tests/test_hydro_pallas.py:209) has an unsourced tolerance |
| Runs on NVIDIA and AMD GPUs | README | GPU-tier bounds measured on A100 and MI300A (e.g. tests/test_cosmo_tophat_collapse.py:157) | shown | public CI is CPU only |
| Weak scaling to 4096 GH200 GPUs | README | measurement, see below | shown | |
| A negative pressure or NaN stops the run (dt = -inf), on one device, across devices and in AMR | CLAUDE.md | tests/test_compute_dt_nan.py:39, :45, :118 | validated | |
| Contradictory settings are refused, not silently replaced | CLAUDE.md | tests/test_run_kind.py:51, tests/test_hydro_cfl_stability.py:74, tests/test_hydro_pallas.py:381 | validated | |
| `field=value` overrides are typed; unknown fields are refused | README | tests/test_cli.py:34, :54 | validated | |
| Each run records its settings and code revision (`run.json`, snapshot headers) | README | tests/test_main_run_dense.py:17, tests/test_provenance.py:56 | validated | |
| Checkpoints round-trip bitwise, also onto another decomposition | docs/RUNNING.md | tests/test_checkpoint_v2.py:40, :62 | validated | |
| A restarted run continues as the uninterrupted one | docs/RUNNING.md | tests/test_checkpoint_v2.py:115 | shown | rtol 1e-2, about 6-60x the measured difference; bit-exact restart not asserted |
| Reference data come from luna-reference-solutions, sha256-checked; a missing file is an error, not a skip | README | tests/test_reference_data.py:19, :26, :33 | validated | |

## Gravity and cosmology

| claim | where stated | test | status | note |
|---|---|---|---|---|
| Periodic self-gravity: Jeans waves against the exact linear solution | README | tests/test_gravity_jeans_wave.py:102 | validated | |
| Isolated boundaries: face potential against a direct sum, force against the uniform sphere | README | tests/test_isolated_gravity.py:94, :108 | validated | |
| Isolated solve sharded equals one device | docs/GRAVITY.md | tests/test_isolated_gravity.py:148 (1e-12, cold and warm) | validated | except the case under known bugs |
| Energy-conserving gravity coupling (`gravity_energy="flux"`, what `auto` resolves to) on the Evrard collapse | README | tests/test_gravity_energy_flux.py:102 | validated | |
| Comoving integration: adiabatic expansion and the EdS clock | README | tests/test_cosmo_adiabatic_expansion.py:59 | validated | |
| Spherical top-hat collapse against the exact closed-FRW solution | README | tests/test_cosmo_tophat_collapse.py:141 | validated | |
| Cosmic web, 512³ gas + 512³ dark matter to z = 0 | README figure, GALLERY | none | shown | |
| Dark-matter power spectrum compared with Arepo on identical initial conditions | README figure, GALLERY | none | shown | figure only (equal on large scales, LUNA's PM lower above k ≈ 5 h/Mpc); an asserted gate is planned. RAMSES is not in this comparison |
| Cross-check with mini-RAMSES and Arepo on adiabatic expansion and top-hat collapse | TEST_PLAN | tests/test_cosmo_a16_xcheck.py:31, :55 | validated against the exact solution | the other codes are printed alongside, not gated |
| Zel'dovich pancake; dark-matter linear growth; coupled gas + DM pancake | GALLERY, docs/GRAVITY.md | tests/test_cosmo.py:26, tests/test_cosmo_dm_growth.py:70, tests/test_cosmo_hydro_coupled.py:84 | shown | regression only (unsourced bounds) |
| External potentials (`g = -grad phi`) | docs/CONFIG.md | tests/test_external_potential.py:22 | validated | |
| Diode outflow boundaries | docs/CONFIG.md | tests/test_outflow_boundary.py:63, :71 | validated | |

## Physics modules

| claim | where stated | test | status | note |
|---|---|---|---|---|
| Radiative cooling tables reproduce published equilibria | README | tests/test_cooling_tables.py:19 | validated | |
| Cooling integrated in the time step | README | tests/test_cooling.py:76, :80 | shown | :76 cannot fail (compares two test-local formulas); :80 asserts a looser bound than its docstring |
| M82-like starburst wind (CC85 injection, Schneider et al. 2018 setup) | README figure, GALLERY | tests/test_wind_first_step.py | shown | first step only; no asserted comparison with CC85 |
| Radiation transport (M1, M2) | README | tests/test_radiation_*.py | experimental | |
| Chemistry networks | README, docs/CHEMISTRY.md | tests/test_chemistry_*.py | experimental | |
| Sink particles; star formation coupled to Torch/AMUSE | README figure, GALLERY | tests/test_sink*.py, tests/test_coupling_*.py | experimental | three coupling tests skip: they import the deleted `sources.sinks.setup_sinks` |
| Block AMR | README figure, docs/AMR.md | tests/test_amr_*.py | experimental | AMR shard equality is asserted to tolerances, not bitwise |
| Gradients through the solver match finite differences | README | tests/test_autodiff_validation.py:67, tests/test_autodiff_sharded.py:165 | validated | for the cases tested |
| Differentiation through whole simulations (the LUNA-word demonstration) | README figure, GALLERY | none for the demonstration | shown | known gaps in docs/AUTODIFF.md; tests/test_autodiff_validation.py:142 passes a known 42 % adjoint mismatch for Neumann gravity |
| White-dwarf / Type Ia detonations | docs/TYPEIA_PLAN.md | not on main | (in development) | |

## Known open bugs (strict xfail tests)

A strict xfail fails the suite when the bug is fixed, so the entry cannot go stale.

- HLLC carbuncles at grid-aligned strong shocks; HLL stays planar:
  tests/test_hydro_hard_carbuncle.py:121 (hllc cases), :149; tests/test_hydro_hard_noh_multid.py:309.
- Keplerian disc with the MC limiter: negative pressure in the fast tail gas inside the
  softening radius after about 20 steps, the run stops (NaN dt):
  tests/test_hydro_hard_keplerian_disc.py:158.
- Isolated gravity, 128³ gathered whole at the default `mg_agg_n=128`, sharded: 2.7e-11 off
  the one-device result: tests/test_gravity_decomposition.py:84.

## Measurements (not tests)

Performance numbers are measurements of one code revision on one machine; they are quoted
with hardware, problem and precision. No other code is timed.

| statement | source |
|---|---|
| Weak scaling, 3D Sedov, 512³ cells per GPU, FP32, JUPITER GH200 (2026-08-06): 2.19 Gcell/s (2193 Mcell/s) on 1 GPU, 7.99 Tcell/s on 4096 GPUs, 88.9 % efficiency. No FP64 point at 4096 GPUs. | docs/GALLERY.md, scaling/jupiter/jupiter_numbers.csv, scaling/jupiter/RESULTS.md (job ids) |
