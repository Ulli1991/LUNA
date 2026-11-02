# What LUNA claims, and the test behind each claim

Statuses:
- **validated**: a test asserts the claim with a gate that does not depend on LUNA's own
  measured error: round-off or an exact identity (bitwise equality, a conservation law, a
  textbook formula), a convergence-order band that contains the scheme's theoretical order,
  an AthenaK matched-method bound (`L1(LUNA) <= L1(AthenaK, same method, same N) + 2 EPS_REF`),
  or published data to its published precision.
- **regression pin**: a test pins LUNA's own result, usually at ~1.3x its measured error
  (docs/TESTING.md, "Hydro accuracy and hard tests"), even when that error is measured
  against an exact solution or another code. A change is caught; correctness is not shown.
- **shown**: real runs, shown in figures or measured; no test asserts the claim. The note
  says what a test does check, if anything.
- **experimental**: under development, not finalised (radiation, chemistry, sinks and the
  Torch coupling, block AMR). These modules have tests; they do not yet meet the standard
  of the hydro core.

Statuses follow the October 2026 test audit and the re-rating of 2026-11-02, each row
re-checked against its test. Test locations are `file:line` of the test function; bound
sources are in each test's docstring or next to its bound table. GPU-only tests and tests
that need 2 or 8 devices run in the GPU and `--devices` tiers, not in the CPU fast tier
(docs/TESTING.md).

## Hydrodynamics

| claim | where stated | test | status | note |
|---|---|---|---|---|
| Euler equations in 1D, 2D, 3D, unsplit MUSCL-Hancock with PLM; second order on smooth flows | README | tests/test_hydro_accuracy_smooth.py:160, tests/test_hydro_accuracy_linear_waves.py:231, tests/test_regressions.py:105 | validated | the order bands contain the theoretical 2 (sound wave L1 1.76-2.16, test_hydro_accuracy_smooth.py:15; linear wave 1.77-2.17, test_hydro_accuracy_linear_waves.py:80); mass to 1e-13; the L1 values themselves are pins |
| HLLC (default) and HLL fluxes match their textbook definitions and Toro's tests | README | tests/test_hydro_accuracy_riemann.py:67, :106, :128 | validated | :106 equals Toro's formulas to 1e-14, :128 the exact flux of an isolated contact to 1e-14; the flux errors on Toro's tests at :67 are pins |
| Shock tubes (Toro 1-5) converge to the exact Riemann solution | README | tests/test_hydro_accuracy_shocktube.py:157 | validated | the L1(rho) order bands contain 1 for Toro 1, 2, 3 and 5 (:63); Toro 4 (0.57-0.97) and the L1/L2/front bounds are pins |
| `riemann_solver="hll"` is the robust fallback: HLL keeps grid-aligned strong shocks planar where HLLC carbuncles | README, GALLERY | tests/test_hydro_hard_carbuncle.py:121 (hll cases pass, hllc cases strict xfail), :143 | regression pin | the planarity bounds are ~1.3x LUNA's HLL result (Quirk row spread 5.5e-7, measured 4.38e-7, :113); HLLC is six orders above (0.72). Chosen by the user, never switched automatically (no automatic switch in src/) |
| Sedov-Taylor blast against the exact solution | README | tests/test_hydro_hard_sedov_offaxis.py:243 (2D cylindrical, off-axis), :219 | regression pin | :219 checks the exact-solution code against Kamm & Timmes 2007 (2e-4); LUNA's metrics are pinned at \|dR - measured\| <= 0.05 cells and 1.1x measured (:161-164); the GPU order is a lower bound 0.78 against measured 0.83 (:198); sharded runs are bitwise (:262) |
| 3D Sedov shock radius and profile | README figure, GALLERY | tests/test_hydro_accuracy_sedov.py:69 | regression pin | scalar checks; the "<1 %" in GALLERY is read off a plot |
| Noh (planar, 2D, 3D) and Guderley implosion against exact solutions | README | tests/test_hydro_hard_noh_planar.py:150, tests/test_hydro_hard_noh_multid.py:286, tests/test_hydro_hard_guderley.py:345 | validated | the L1 order bands contain the theoretical 1: planar (test_hydro_hard_noh_planar.py:43-52), 2D (test_hydro_hard_noh_multid.py:279-281), Guderley at R_s 0.8 (test_hydro_hard_guderley.py:224); 3D Noh is one N, a pin; the Guderley solution is checked against RKB Table 1 and ExactPack |
| Linear growth of KH and Rayleigh-Taylor against linear theory | README | tests/test_hydro_accuracy_kh.py:87, tests/test_hydro_hard_rayleigh_taylor.py:191 | validated | the order bands contain 2: KH L1 1.91-2.31 (test_hydro_accuracy_kh.py:15), RT hllc-minmod L1 1.61-2.01 (test_hydro_hard_rayleigh_taylor.py:85); hllc-mc RT (2.73-3.13) and the L1 values are pins |
| Linear growth of Richtmyer-Meshkov against linear theory | README | tests/test_hydro_hard_richtmyer_meshkov.py:218 | regression pin | 1.3x measured, orders 0.4-1.05 with no theoretical order (:22-58); only the theory code is checked against published worked examples (1e-9) and the Bromwich inversion (1e-12) |
| Nonlinear KH (McNally et al. 2012) against the Pencil Code and AthenaK | README | tests/test_hydro_hard_mcnally_kh.py:288, :457, :468 | regression pin | M(t) against the Pencil curve and rho, V_y against AthenaK PPMX 4096² are 1.3x measured; the matched bound (:468, GPU) is L1 <= PLM_RATIO x L1(AthenaK PLM) + 2 EPS_REF with PLM_RATIO = 1.05 set from LUNA's measured 4 % excess (:433), so it contains a LUNA number; LUNA van Leer is within 5 % of AthenaK PLM, 3-4 % above it in V_y (not "no less accurate") |
| KH at 8192² next to AthenaK | README figure, GALLERY | none | shown | accuracy comparison by eye, not an asserted gate; GALLERY's "converged to AthenaK" is stronger than the evidence |
| Shu-Osher, Woodward-Colella, Titarev-Toro | GALLERY | tests/test_hydro_hard_shu_osher.py:137, tests/test_hydro_hard_woodward_colella.py:156, tests/test_hydro_hard_titarev_toro.py:136; pins at :129, :148, :128 | validated | matched method: LUNA van Leer against AthenaK PLM at the same N, L1 <= L1(AthenaK PLM) + 2 EPS_REF; EPS_REF, the reference's own error, is the larger of AthenaK's self-convergence and its distance to LUNA's fine run, which can only loosen the bound; first order fails it in Shu-Osher (teeth, test_hydro_hard_shu_osher.py:146). The L1/L2 bounds against AthenaK PPMX + RK4 at 16384 / 16384 / 65536 cells are pins |
| Liska-Wendroff implosion (early time), Kurganov-Tadmor 2D Riemann configuration 3 | TEST_PLAN | tests/test_hydro_hard_implosion.py:410, tests/test_hydro_hard_kt_riemann.py:446 (matched method, GPU); pins at test_hydro_hard_implosion.py:400, test_hydro_hard_kt_riemann.py:291, :298 | validated | matched method as above against AthenaK PPMX 6400² (implosion t = 0.045) and 8192² (KT 3 outside the core); diagonal symmetry bitwise, flux balance 1e-14; implosion late time, KT cores and KT configuration 6 are pins (no converged reference in either code) |
| Cloud-crushing blob (Agertz et al.) | TEST_PLAN | tests/test_hydro_hard_agertz_blob.py:109 | regression pin | |
| Polytrope in self-gravity held at rest | TEST_PLAN | tests/test_hydro_hard_well_balanced.py:110 | regression pin | the scheme is not well balanced (5-23 % central drift over 10 crossings); the bound followed the code four times (:7-10) |
| Driven turbulence (OU forcing) | README, GALLERY | tests/test_turb_spectrum.py:43 (initial spectra); tests/test_ou_forcing_split.py:71, :95 | shown | the forcing-split tests never run the forcing stage |

## Devices, kernels, runs

| claim | where stated | test | status | note |
|---|---|---|---|---|
| A run split over several devices gives the single-device result | README, CLAUDE.md | per module in the table below | validated | bitwise for the hydro step; to stated round-off tolerances for gravity, radiation, autodiff, walls and floors; AMR, chemistry and cosmology/PM differ (below) |
| Pallas hydro kernel is the XLA kernel | README | tests/test_hydro_pallas_matrix.py:37 (bitwise on GPU, every dim/limiter/solver/precision), tests/test_hydro_pallas.py:403 | validated | the full production step on GPUs (tests/test_hydro_pallas.py:209) has an unsourced tolerance |
| Runs on NVIDIA and AMD GPUs | README | GPU-tier bounds measured on A100 and MI300A (e.g. tests/test_cosmo_tophat_collapse.py:157) | shown | public CI is CPU only; the committed Raven GPU reports are failing (docs/TESTING.md) |
| Weak scaling to 4096 GH200 GPUs | README | measurement, see below | shown | |
| A negative pressure or NaN stops the run (dt = -inf), on one device, across devices and in AMR | CLAUDE.md | tests/test_compute_dt_nan.py:39, :45, :118 | validated | |
| Contradictory settings are refused, not silently replaced | CLAUDE.md | tests/test_run_kind.py:51, tests/test_hydro_cfl_stability.py:74, tests/test_hydro_pallas.py:381 | validated | |
| `field=value` overrides are typed; unknown fields are refused | README | tests/test_cli.py:34, :54 | validated | |
| Each run records its settings and code revision (`run.json`, snapshot headers) | README | tests/test_main_run_dense.py:17, tests/test_provenance.py:56, :75 | validated | |
| Checkpoints round-trip bitwise, also onto another decomposition | docs/RUNNING.md | tests/test_checkpoint_v2.py:40, :62 | validated | |
| A restarted run continues as the uninterrupted one | docs/RUNNING.md | tests/test_checkpoint_v2.py:115 | shown | rtol 1e-2, about 6-60x the measured difference; bit-exact restart not asserted |
| Reference data come from luna-reference-solutions at the commit in `reference-solutions.lock`, sha256-checked; a missing file is an error, not a skip | README | tests/test_reference_data.py:56, :62, :69, :78, :93, :102, :112, :123 | validated | another commit or a re-generated MANIFEST.json is refused; `LUNA_REFERENCE_SOLUTIONS_UNLOCKED=1` is logged |

### Sharded equals one device, per module

| module | test | asserted |
|---|---|---|
| hydro | tests/test_hydro_hard_sedov_offaxis.py:262 (2x4 mesh) | bitwise, max \|diff\| = 0 (:269) |
| hydro | tests/test_hydro_hard_mcnally_kh.py:318 | bitwise (SHARDED = 0.0, :244) |
| hydro | tests/test_hydro_hard_implosion.py:199, tests/test_hydro_hard_kt_riemann.py:335 | bitwise (:206; SHARD = 0.0, :107) |
| hydro, Pallas | tests/test_hydro_pallas.py:403 (GPU) | bitwise, `assert_array_equal` (:417) |
| hydro | tests/test_seam_consistency.py:116; tests/test_hydro_accuracy_shocktube.py:217 | rel < 1e-12 (:155); rtol 1e-12, atol 0 (:224) |
| walls, diode, floors | tests/test_hydro_wall_conservation.py:49; tests/test_outflow_boundary.py:63; tests/test_floor_conservation.py:111 | rel <= 1e-12 (:57); < 1e-12 (:68); rtol 1e-14, atol 1e-16 (:121) |
| gravity, periodic and isolated | tests/test_gravity_decomposition.py:61, :82, :105 | rel < 2.5e-12 (TOL, :21) |
| gravity, isolated | tests/test_isolated_gravity.py:148 | < 1e-12, cold and warm (:164) |
| gravity, flux coupling | tests/test_gravity_energy_flux.py:305 | `array_equal` and rel < 1e-14 (:340, :344) |
| cosmology / PM | none | no sharded run is compared with a one-device run; only the PM bridge field (tests/test_gravity_decomposition.py:105) |
| radiation | tests/test_radiation_coupled.py:149; tests/test_radiation_m2_stage.py:220 | rtol 1e-12, atol 1e-14 (:168); rtol 1e-11, atol 1e-13 (:251) |
| chemistry | tests/test_chemistry_integrator.py:296 | rtol 1e-8 (:315); the chemistry files currently run in no standard config |
| AMR | tests/test_amr_shard.py:22; tests/test_amr_multiprocess.py:54 | atol 1e-10 with `allclose`'s default rtol 1e-5 (:35, :41); two processes against one, both on 2 devices, bitwise (:69) |
| autodiff | tests/test_autodiff_sharded.py:123 | rel < 1e-11 (:129) |

## Gravity and cosmology

| claim | where stated | test | status | note |
|---|---|---|---|---|
| Periodic self-gravity: Jeans waves against the exact linear solution | README | tests/test_gravity_jeans_wave.py:102 | validated | the minmod order bands contain 2 (1.98-2.38 etc., :20-40); hllc-mc (3.00-3.40) and the L1 values are pins |
| Isolated boundaries: face potential against a direct sum, force against the uniform sphere | README | tests/test_isolated_gravity.py:94, :108 | regression pin | round bounds 1.5e-2 (measured 8.0e-3, :105) and 1e-2 (measured 6.1e-3, "the l <= 2 truncation", :127), not derived |
| Isolated solve sharded equals one device | docs/GRAVITY.md | tests/test_isolated_gravity.py:148, tests/test_gravity_decomposition.py:82 | validated | 1e-12 and 2.5e-12, cold and warm; the whole-grid gathered case is fixed (fixed-order `tree_sum`) |
| Energy-conserving gravity coupling (`gravity_energy="flux"`, what `auto` resolves to) on the Evrard collapse | README | tests/test_gravity_energy_flux.py:136 | validated | \|dE\|/\|E0\| <= 5.7e-5 at solve tolerance 1e-4 and <= 5.8e-11 at 1e-10 (:31): below the tolerance and falling with it, so the error is the solve's; the per-time bounds of :113 are pins |
| Comoving integration: adiabatic expansion and the EdS clock | README | tests/test_cosmo_adiabatic_expansion.py:59, :69 | validated | :59 at round-off (1e-12, 1e-13); the clock at :69 is a pin (1.3x measured 4.45e-6) |
| Spherical top-hat collapse against the exact closed-FRW solution | README | tests/test_cosmo_tophat_collapse.py:141 | regression pin | CORE/SLOPE bounds are 1.3x measured (:117-123); the exact solution itself is checked to 1e-9 and momentum to 1e-13; 1 % stronger gravity fails (teeth) |
| Cosmic web, 512³ gas + 512³ dark matter to z = 0 | README figure, GALLERY | none | shown | |
| Dark-matter power spectrum compared with Arepo on identical initial conditions | README figure, GALLERY | none | shown | figure only (equal on large scales, LUNA's PM lower above k ≈ 5 h/Mpc); an asserted gate is planned. RAMSES is not in this comparison |
| Cross-check with mini-RAMSES and Arepo on adiabatic expansion and top-hat collapse | TEST_PLAN | tests/test_cosmo_a16_xcheck.py:31, :55 | validated against the exact solution (:31); regression pin (:55) | the other codes are printed alongside, never gated |
| Zel'dovich pancake; dark-matter linear growth; coupled gas + DM pancake | GALLERY, docs/GRAVITY.md | tests/test_cosmo.py:26, tests/test_cosmo_dm_growth.py:70, tests/test_cosmo_hydro_coupled.py:84 | regression pin | unsourced bounds |
| External potentials (`g = -grad phi`) | docs/CONFIG.md | tests/test_external_potential.py:22 | validated | |
| Diode outflow boundaries | docs/CONFIG.md | tests/test_outflow_boundary.py:63, :71 | validated | |

## Physics modules

| claim | where stated | test | status | note |
|---|---|---|---|---|
| Radiative cooling tables reproduce published equilibria | README | tests/test_cooling_tables.py:42, :53 | validated | :42: the Koyama-Inutsuka 2002 fit (low-T coefficient 1.4e-2) balances the published states to their published rounding (756 K: 0.14 % off, bound 0.21 %; 1500 K: 2.0 % off, bound 27 %, n published to one digit, so a weak check); the printed 14 fails. :53: both tables reproduce the fit to the a-priori log-log interpolation bound; a 1 % heating error fails |
| Cooling integrated in the time step | README | tests/test_cooling.py:76, :80 | shown | :76 cannot fail (compares two test-local formulas); :80 asserts a looser bound than its docstring |
| M82-like starburst wind (CC85 injection, Schneider et al. 2018 setup) | README figure, GALLERY | tests/test_wind_first_step.py | shown | first step only; no asserted comparison with CC85 |
| Radiation transport (M1, M2) | README | tests/test_radiation_*.py | experimental | |
| Chemistry networks | README, docs/CHEMISTRY.md | tests/test_chemistry_*.py | experimental | excluded on GitHub (tables in `$LUNA_DATA`, tests/ci_exclude.json) and a collection error in the committed Raven report: they run in no standard config yet |
| Sink particles; star formation coupled to Torch/AMUSE | README figure, GALLERY | tests/test_sink*.py, tests/test_coupling_*.py | experimental | the coupling tests inject sinks directly since `sources.sinks.setup_sinks` was deleted (tests/test_coupling_sink_to_star.py:36-41); the two older setup_sinks files are parked in docs/specs/sinks/, outside `testpaths` |
| Block AMR | README figure, docs/AMR.md | tests/test_amr_*.py | experimental | shard equality to tolerances (tests/test_amr_shard.py:22), two processes bitwise (tests/test_amr_multiprocess.py:54) |
| Gradients through the solver match finite differences | README | tests/test_autodiff_validation.py:67, tests/test_autodiff_sharded.py:165 | validated | for the cases tested: best central difference within a fixed 1e-6 (test_autodiff_validation.py:95), not a fitted bound |
| Differentiation through whole simulations (the LUNA-word demonstration) | README figure, GALLERY | none for the demonstration | shown | known gaps in docs/AUTODIFF.md; tests/test_autodiff_validation.py:142 passes a known 42 % adjoint mismatch for Neumann gravity |
| White-dwarf / Type Ia detonations | docs/TYPEIA_PLAN.md | not on main | (in development) | |

## Known open bugs (strict xfail tests)

A strict xfail fails the suite when the bug is fixed, so the entry cannot go stale.

- HLLC carbuncles at grid-aligned strong shocks; HLL stays planar:
  tests/test_hydro_hard_carbuncle.py:121 (hllc cases), :149; tests/test_hydro_hard_noh_multid.py:309.
- Keplerian disc with the MC limiter: negative pressure in the fast tail gas inside the
  softening radius after about 20 steps, the run stops (NaN dt):
  tests/test_hydro_hard_keplerian_disc.py:158.

## Measurements (not tests)

Performance numbers are measurements of one code revision on one machine; they are quoted
with hardware, problem and precision. No other code is timed.

| statement | source |
|---|---|
| Weak scaling, 3D Sedov, 512³ cells per GPU, FP32, JUPITER GH200 (2026-08-06): 2.19 Gcell/s (2193 Mcell/s) on 1 GPU, 7.99 Tcell/s on 4096 GPUs, 88.9 % efficiency. No FP64 point at 4096 GPUs. | docs/GALLERY.md, scaling/jupiter/jupiter_numbers.csv, scaling/jupiter/RESULTS.md (job ids) |
